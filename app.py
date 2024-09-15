import streamlit as st
import groq
import os
import json
import time
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Generator, Union
import logging
import traceback
from openai import OpenAI
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 10
TEMPERATURE = 0.2
STEP_MAX_TOKENS = 2000
FINAL_ANSWER_MAX_TOKENS = 2000
ONE_SHOT_MAX_TOKENS = 2000

# Provider and model configurations
PROVIDERS = {
    "Groq": {
        "client": groq.Groq(),
        "models": [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "gemma-7b-it"
        ]
    },
    "OpenRouter": {
        "client": OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        ),
        "models": [
            "nousresearch/hermes-3-llama-3.1-405b:free",
            "mistralai/mistral-7b-instruct:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "microsoft/phi-3-medium-128k-instruct:free",
            "meta-llama/llama-3-8b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "huggingfaceh4/zephyr-7b-beta:free",
            "mistralai/pixtral-12b:free",
            "qwen/qwen2-vl-7b-instruct:free",
            "qwen/qwen2-7b-instruct:free",
            "google/gemma-2-9b:free",
            "openchat/openchat-3.5-7b:free"
        ]
    }
}

@st.cache_resource
def get_provider_client(provider: str):
    return PROVIDERS[provider]["client"]

def make_api_call(provider: str, model: str, messages: List[Dict[str, str]], max_tokens: int, is_final_answer: bool = False) -> Dict[str, Union[str, Dict]]:
    """
    Make an API call to the selected provider's LLM model with retry logic.

    Args:
        provider (str): The selected provider (Groq or OpenRouter).
        model (str): The selected model.
        messages (List[Dict[str, str]]): List of message dictionaries for the conversation.
        max_tokens (int): Maximum number of tokens for the response.
        is_final_answer (bool): Whether this is the final answer call.

    Returns:
        Dict[str, Union[str, Dict]]: Parsed response from the API or error information.
    """
    client = get_provider_client(provider)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=TEMPERATURE
            )
            content = response.choices[0].message.content
            
            # Try to parse as JSON, but fallback to plain text if it fails
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"content": content, "error": None}
        except Exception as e:
            logging.error(f"API call attempt {attempt + 1} failed: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                error_message = f"Failed to generate response after {MAX_RETRIES} attempts."
                logging.error(error_message)
                return {"error": f"{error_message} Error: {str(e)}"}
            time.sleep(RETRY_DELAY)

def generate_response(provider: str, model: str, prompt: str) -> Generator[Tuple[List[Tuple[str, str, float]], float], None, None]:
    """
    Generate a step-by-step response to the given prompt using the selected provider and model.

    Args:
        provider (str): The selected provider.
        model (str): The selected model.
        prompt (str): The user's input prompt.

    Yields:
        Tuple[List[Tuple[str, str, float]], float]: A tuple containing the list of steps and total thinking time.
    """
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "I will now think step by step, starting with decomposing the problem."}
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        start_time = time.time()
        step_data = make_api_call(provider, model, messages, STEP_MAX_TOKENS)
        thinking_time = time.time() - start_time
        total_thinking_time += thinking_time
        
        if "error" in step_data and step_data["error"]:
            steps.append((f"Error in Step {step_count}", step_data["error"], thinking_time))
            break
        
        content = step_data.get("content", "")
        title = f"Step {step_count}"
        if "title" in step_data:
            title += f": {step_data['title']}"
        
        steps.append((title, content, thinking_time))
        messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
        next_action = step_data.get("next_action", "final_answer")
        if next_action == "final_answer":
            break
        
        step_count += 1
        yield steps, None

    # Generate final answer
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})
    
    start_time = time.time()
    final_data = make_api_call(provider, model, messages, FINAL_ANSWER_MAX_TOKENS, is_final_answer=True)
    thinking_time = time.time() - start_time
    total_thinking_time += thinking_time
    
    if "error" in final_data and final_data["error"]:
        steps.append(("Error in Final Answer", final_data["error"], thinking_time))
    else:
        steps.append(("Final Answer", final_data.get("content", ""), thinking_time))
    
    yield steps, total_thinking_time

def get_system_prompt() -> str:
    """
    Return the system prompt for the LLM.

    Returns:
        str: The system prompt.
    """
    return """You are an expert AI assistant that explains your reasoning step by step. For each step, 
    provide a title that describes what you're doing in that step, along with the content. 
    Decide if you need another step or if you're ready to give the final answer. 
    USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. 
    IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. 
    CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. 
    FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. 
    DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.

    Format your response as follows:
    {
        "title": "[Step Title]",
        "content": "[Step Content]",
        "next_action": "[continue/final_answer]"
    }
    """

def get_one_shot_response(provider: str, model: str, prompt: str) -> str:
    """
    Generate a one-shot response without multi-turn COT reasoning.

    Args:
        provider (str): The selected provider.
        model (str): The selected model.
        prompt (str): The user's input prompt.

    Returns:
        str: The one-shot response from the model.
    """
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]
    
    response = make_api_call(provider, model, messages, ONE_SHOT_MAX_TOKENS)
    return response.get('content', 'Error: Unable to generate one-shot response')

def log_session(provider: str, model: str, prompt: str, cot_steps: List[Tuple[str, str, float]], one_shot_response: str):
    """
    Log the session details, including all requests, reasoning steps, and responses.

    Args:
        provider (str): The selected provider.
        model (str): The selected model.
        prompt (str): The user's input prompt.
        cot_steps (List[Tuple[str, str, float]]): List of COT reasoning steps.
        one_shot_response (str): The one-shot response.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_log_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Session Log - {timestamp}\n\n")
        f.write(f"Provider: {provider}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Prompt: {prompt}\n\n")
        
        f.write("Chain-of-Thought Reasoning:\n")
        for step_title, step_content, thinking_time in cot_steps:
            f.write(f"{step_title}\n")
            f.write(f"Content: {step_content}\n")
            f.write(f"Thinking Time: {thinking_time:.2f} seconds\n\n")
        
        f.write("One-Shot Response:\n")
        f.write(f"{one_shot_response}\n")

def main():
    """
    Main function to set up and run the Streamlit app.
    """
    st.set_page_config(page_title="g1 prototype", page_icon="🧠", layout="wide")
    
    st.title("g1: Multi-Provider o1-like Reasoning Chains with Comparison")
    
    st.markdown("""
    This advanced prototype uses prompting strategies to improve LLM reasoning capabilities through o1-like reasoning chains. 
    It allows the LLM to "think" and solve logical problems that usually stump leading models. 
    You can now choose between different providers and models to compare their performance, including a one-shot response for comparison.
                
    Open source [repository here](https://github.com/bklieger-groq)
    """)
    
    # Sidebar for provider and model selection
    with st.sidebar:
        st.header("Model Selection")
        provider = st.selectbox("Choose Provider", list(PROVIDERS.keys()))
        model = st.selectbox("Choose Model", PROVIDERS[provider]["models"])
        
        st.header("Options")
        show_thinking_time = st.checkbox("Show thinking time for each step", value=True)
        show_total_tokens = st.checkbox("Show total tokens used", value=False)
        log_session_option = st.checkbox("Log session to file", value=False)
    
    # Multi-line text input for user query
    user_query = st.text_area("Enter your query:", placeholder="e.g., How many 'R's are in the word strawberry?", height=100)
    
    # Send button
    if st.button("Send", key="send_button") or (user_query and user_query.endswith('\n')):
        if user_query:
            st.write("Generating responses...")
            
            # Create empty elements to hold the generated text and total time
            cot_container = st.empty()
            time_container = st.empty()
            token_container = st.empty()
            one_shot_container = st.empty()
            
            try:
                # Generate and display the COT response
                cot_steps = []
                total_thinking_time = 0
                for steps, thinking_time in generate_response(provider, model, user_query):
                    cot_steps = steps
                    total_thinking_time = thinking_time
                    
                    with cot_container.container():
                        st.subheader("Chain-of-Thought Reasoning:")
                        for i, (title, content, step_time) in enumerate(steps):
                            with st.expander(title, expanded=True):
                                st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                                if show_thinking_time:
                                    st.text(f"Thinking time: {step_time:.2f} seconds")
                
                # Show total time
                if total_thinking_time is not None:
                    time_container.markdown(f"**Total thinking time: {total_thinking_time:.2f} seconds**")
                    
                    if show_total_tokens:
                        # Estimate token usage (this is a rough estimate)
                        total_tokens = sum(len(step[1].split()) for step in cot_steps) * 1.3  # Multiply by 1.3 as a rough token-to-word ratio
                        token_container.markdown(f"**Estimated total tokens used: {int(total_tokens)}**")
                
                # Generate and display the one-shot response
                one_shot_response = get_one_shot_response(provider, model, user_query)
                with one_shot_container.container():
                    st.subheader("One-Shot Response (without multi-turn COT):")
                    st.write(one_shot_response)
                
                # Log session if option is selected
                if log_session_option:
                    log_session(provider, model, user_query, cot_steps, one_shot_response)
                    st.success("Session logged successfully!")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logging.error(f"Error in main loop: {str(e)}")
                logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
