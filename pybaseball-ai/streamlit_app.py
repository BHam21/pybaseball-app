import streamlit as st
from typing import Generator
import re
import json
import os


# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(page_icon="💬", layout="wide", page_title="LLM & Baseball Agent Dashboard")


# --- IMPORTS FOR GROQ CHAT ---
from groq import Groq

# --- IMPORTS FOR CREWAI AGENTS ---
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import CodeInterpreterTool
from pydantic import BaseModel, Field

# --- PYBASEBALL IMPORT ---
import pybaseball  # (this will be used by the generated code)

# --- HELPER FUNCTION FOR ICON ---
def icon(emoji: str):
    """Display a large emoji icon."""
    st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

# --- PAGE HEADER ---
icon("🏎️")
st.title("LLM & Baseball Agent Dashboard")
st.write("Switch between a Groq-based chat interface and a Crew-powered baseball data agent in the sidebar.")

# --- SIDEBAR MODE SELECTION ---
mode = st.sidebar.radio("Select Mode", options=["Chat Mode", "Baseball Agent Mode"])

# =============================================================================
#                           CHAT MODE (Groq Chat)
# =============================================================================
if mode == "Chat Mode":
    st.subheader("Groq Chat Streamlit App")
    
    # --- Initialize Groq Client ---
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    # --- Session State for Chat ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    # --- Model Details ---
    models = {
        "gemma2-9b-it": {"name": "Gemma2-9b-it", "tokens": 8192, "developer": "Google"},
        "llama-3.3-70b-versatile": {"name": "LLaMA3.3-70b-versatile", "tokens": 128000, "developer": "Meta"},
        "llama-3.1-8b-instant": {"name": "LLaMA3.1-8b-instant", "tokens": 128000, "developer": "Meta"},
        "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
        "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
        "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
    }

    # --- Layout for Model Selection and max_tokens ---
    col1, col2 = st.columns(2)
    with col1:
        model_option = st.selectbox(
            "Choose a model:",
            options=list(models.keys()),
            format_func=lambda x: models[x]["name"],
            index=4  # default to mixtral
        )
    # --- Reset chat history if model changes ---
    if st.session_state.selected_model != model_option:
        st.session_state.messages = []
        st.session_state.selected_model = model_option
    max_tokens_range = models[model_option]["tokens"]
    with col2:
        max_tokens = st.slider(
            "Max Tokens:",
            min_value=512,
            max_value=max_tokens_range,
            value=min(32768, max_tokens_range),
            step=512,
            help=f"Adjust the maximum number of tokens (max for selected model: {max_tokens_range})"
        )

    # --- Display Chat History ---
    for message in st.session_state.messages:
        avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
        """Yield chat response content from the Groq API response."""
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # --- Chat Input & Response Generation ---
    if prompt := st.chat_input("Enter your prompt here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar='👨‍💻'):
            st.markdown(prompt)
        try:
            chat_completion = client.chat.completions.create(
                model=model_option,
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                max_tokens=max_tokens,
                stream=True
            )
            with st.chat_message("assistant", avatar="🤖"):
                chat_responses_generator = generate_chat_responses(chat_completion)
                full_response = st.write_stream(chat_responses_generator)
        except Exception as e:
            st.error(e, icon="🚨")
        if isinstance(full_response, str):
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            combined_response = "\n".join(str(item) for item in full_response)
            st.session_state.messages.append({"role": "assistant", "content": combined_response})

# =============================================================================
#                      BASEBALL AGENT MODE (Crew Pipeline)
# =============================================================================
elif mode == "Baseball Agent Mode":
    st.subheader("Baseball Data & Visualization Agent")
    st.write("Enter a query about baseball data (e.g., 'Show me the batting averages for the 2021 Yankees') and the agent will generate code to pull data using pybaseball and visualize it.")

    # --- Define the Pydantic model for structured query output ---
    class BaseballQueryOutput(BaseModel):
        data_function: str = Field(
            ...,
            description="The pybaseball function to call (e.g., 'batting_stats', 'pitching_stats')."
        )
        parameters: dict = Field(
            ...,
            description="Parameters for the pybaseball function call (e.g., {'year': 2021, 'team': 'NYY'})."
        )
        visualization: str = Field(
            ...,
            description="Instructions for the type of visualization (e.g., 'bar chart', 'line plot')."
        )

    # --- Setup LLM for Crew Agents ---
    llm = LLM(
        model="sambanova/DeepSeek-R1-Distill-Llama-70B",
        temperature=0.7
    )

    # --- Query Parsing Agent ---
    query_parser_agent = Agent(
        role="Baseball Data Analyst",
        goal="Extract baseball data retrieval details from this user query: {query}.",
        backstory="You are a baseball data analyst specialized in retrieving and analyzing baseball statistics using pybaseball.",
        llm=llm,
        verbose=True,
        memory=True,
    )
    query_parsing_task = Task(
        description="Analyze the user query and extract baseball data retrieval details.",
        expected_output="A dictionary with keys: 'data_function', 'parameters', 'visualization'.",
        output_pydantic=BaseballQueryOutput,
        agent=query_parser_agent,
    )

    # --- Code Writer Agent ---
    code_writer_agent = Agent(
        role="Senior Python Developer",
        goal="Write Python code that retrieves baseball data using pybaseball and generates a visualization as instructed.",
        backstory="""You are a Senior Python developer with expertise in pybaseball and data visualization libraries 
                    (such as matplotlib, seaborn, or plotly). Your code should be production-ready and well commented.""",
        llm=llm,
        verbose=True,
    )
    code_writer_task = Task(
        description="Write Python code to retrieve baseball data using pybaseball and visualize it based on the extracted details.",
        expected_output="A clean and executable Python script file (.py) for baseball data retrieval and visualization.",
        agent=code_writer_agent,
    )

    # --- Code Execution Agent (with Code Interpreter Tool) ---
    code_interpreter_tool = CodeInterpreterTool()
    code_execution_agent = Agent(
        role="Senior Code Execution Expert",
        goal="Review and execute the generated Python code to pull baseball data and generate the visualization.",
        backstory="You are an expert in executing Python code and ensuring that the code runs properly to produce the intended output.",
        tools=[code_interpreter_tool],
        allow_code_execution=True,
        llm=llm,
        verbose=True,
    )
    code_execution_task = Task(
        description="Review and execute the generated Python code to retrieve baseball data and visualize it.",
        expected_output="A clean and executable Python script file (.py) for baseball data retrieval and visualization.",
        agent=code_execution_agent,
    )

    # --- Assemble the Crew (Sequential Process) ---
    crew = Crew(
        agents=[query_parser_agent, code_writer_agent, code_execution_agent],
        tasks=[query_parsing_task, code_writer_task, code_execution_task],
        process=Process.sequential
    )

    # --- Baseball Query Input & Pipeline Execution ---
    baseball_query = st.text_input("Enter your baseball data query:")
    if st.button("Run Baseball Agent") and baseball_query:
        with st.spinner("Processing your query..."):
            # 1) Query Parsing Task
            try:
                parsed_output = query_parsing_task.execute(query=baseball_query)
                st.write("**Parsed Query Details:**")
                st.json(parsed_output)
            except Exception as e:
                st.error(f"Error during query parsing: {e}")
                parsed_output = None

            if parsed_output:
                # 2) Code Writing Task
                try:
                    code_script = code_writer_task.execute(input=parsed_output)
                    st.write("**Generated Code:**")
                    st.code(code_script, language="python")
                except Exception as e:
                    st.error(f"Error during code generation: {e}")
                    code_script = None

                # 3) Code Execution Task
                if code_script:
                    try:
                        execution_output = code_execution_task.execute(input=code_script)
                        st.write("**Execution Output:**")
                        st.write(execution_output)
                    except Exception as e:
                        st.error(f"Error during code execution: {e}")