import streamlit as st
from langchain_groq import ChatGroq
# Use standard langchain imports where possible for stability
from langchain_classic.chains import LLMMathChain, LLMChain
from langchain_classic.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
# Fix: Import AgentType and initialize_agent correctly
from langchain_classic.agents import AgentType, Tool, initialize_agent
import os
from dotenv import load_dotenv
from langchain_classic.callbacks import StreamlitCallbackHandler

load_dotenv()

## Setting up the streamlit app
st.set_page_config(page_title="MathsGPT: Problem Solver & Data Assistant", page_icon="🧮")
st.title("MathsGPT with Llama-3.1")

# Ensure API Key is available
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables.")
    st.stop()

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

## --- Initializing Tools ---

# 1. Wikipedia Tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia Search",
    description="Useful for answering questions about current events or general facts.",
    func=wikipedia_wrapper.run,
)

# 2. Calculator Tool (Fixing the Prompting Issue)
# We use a strict prompt to ensure the LLM only output code for the math chain
math_template = """Translate a math problem into a Python expression that can be executed by numexpr.
Only return the code, no text.
Question: {question}"""

math_prompt = PromptTemplate(input_variables=["question"], template=math_template)
math_chain = LLMMathChain.from_llm(llm=llm, prompt=math_prompt, verbose=True)

calculator = Tool(
    name="Calculator",
    description="Tool for answering math questions. Input must be a clear mathematical expression.",
    func=math_chain.run,
)

# 3. Reasoning Tool
# Fixed: input_variables (not input_variavles)
reasoning_prompt = """You are a helpful mathematical assistant. 
Explain your reasoning clearly and provide a step-by-step solution.
Question: {question}
Answer:"""

prompt_template = PromptTemplate(
    input_variables=['question'],  # Fixed typo here
    template=reasoning_prompt
)

reasoning_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

reasoning_tool = Tool(
    name="Reasoning tool",
    description="Use this for word problems that require logical reasoning or multi-step math.",
    func=reasoning_chain.run,
)

## --- Initialize Agent ---

assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True, # Critical for handling 'Unknown format' errors
)

# Session State Management
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm MathsGPT! How can I help you with numbers today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

## --- User Interaction ---
question = st.text_area("Enter your question:", 
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. How many fruits do I have left?")

if st.button("Find My Answer"):
    if question:
        with st.spinner("Calculating..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            # Define Streamlit callback for 'thought' visibility
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            
            # Execute agent
            try:
                # Pass only the latest question string to .run() for legacy agents
                response = assistant_agent.run(question, callbacks=[st_cb])
                
                st.session_state.messages.append({'role': 'assistant', "content": response})
                st.write('### Response:')
                st.success(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")