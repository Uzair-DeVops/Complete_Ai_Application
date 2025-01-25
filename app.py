import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import os
import tempfile
import time
from pathlib import Path

import os
import google.generativeai as genai
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import tempfile
import time
from pathlib import Path


# Load environment variables
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service.json"

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key="AIzaSyDlGuiJOqQePVsQEu5gWiftb74RDGvcq-c")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY)

# Define Tools
def get_available_tools():
    from tools import llm_fallback, calculator, get_latest_news, get_movie_details, get_recipe, get_distance, get_stock_price, get_ip_address, get_disk_usage, get_time_in_timezone, get_weather
    return [
        calculator,
        get_latest_news,
        get_movie_details,
        get_recipe,
        get_distance,
        get_stock_price,
        get_ip_address,
        get_weather,
        get_time_in_timezone,
        get_disk_usage,
        llm_fallback,
    ]

# Initialize Agent for Tools
tools = get_available_tools()
tool_agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)

def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

# Initialize the video summarizer agent
multimodal_Agent = initialize_agent()




# PDF Processing Functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                if page.extract_text():
                    text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, "answer is not available in the provided context." Do not provide a wrong answer.
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

def user_input_pdf(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]




# Function to get available tools
def get_available_tools():
    from tools import llm_fallback, calculator, get_latest_news, get_movie_details, get_recipe, get_distance, get_stock_price, get_ip_address, get_disk_usage, get_time_in_timezone, get_weather
    return {
        "Calculator": calculator,
        "Latest News": get_latest_news,
        "Movie Details": get_movie_details,
        "Recipe Finder": get_recipe,
        "Distance Calculator": get_distance,
        "Stock Price Checker": get_stock_price,
        "IP Address Finder": get_ip_address,
        "Weather Info": get_weather,
        "Timezone Checker": get_time_in_timezone,
        "Disk Usage Info": get_disk_usage,
        "llm_fallback": llm_fallback,

    }

tools = get_available_tools()

    


# Streamlit UI Setup
st.set_page_config(page_title="AI Application", layout="wide")
st.title("Tool_calling, Rag, video Summarizer Application Powered BY Gemini") 
st.sidebar.title("Options")

# Sidebar: File Upload or Tool Selection
file_type = st.sidebar.radio("Choose an option", ["PDF", "Video", "Tool Interaction"])

if file_type == "PDF":
    pdf_docs = st.sidebar.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])
    if st.sidebar.button("Process PDF") and pdf_docs:
        with st.spinner("Processing PDF..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("PDF processed and indexed.")

    if pdf_docs:
        user_query = st.text_input("Ask a question about the uploaded PDF")
        if user_query:
            with st.spinner("Fetching answer..."):
                response = user_input_pdf(user_query)
                st.write("Answer:", response)

elif file_type == "Video":
    video_file = st.sidebar.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'])
    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name
        st.video(video_path, format="video/mp4")

    if video_file:
        with st.form(key='video_form'):
                    # Text area for user query about the video
                    user_query_video = st.text_area("What insights are you seeking from the video?")

                    # Button to analyze the video
                    submit_button = st.form_submit_button("🔍 Analyze Video")

                    if submit_button:
                        if not user_query_video:
                            st.warning("Please enter a question or insight to analyze the video.")
                        else:
                            try:
                                with st.spinner("Processing video and gathering insights..."):
                                    # Process the video for analysis
                                    processed_video = upload_file(video_path)
                                    while processed_video.state.name == "PROCESSING":
                                        time.sleep(1)
                                        processed_video = get_file(processed_video.name)

                                    # Prompt generation for video analysis
                                    analysis_prompt = f"Analyze the uploaded video for content and context. Respond to the following query using video insights: {user_query_video}"

                                    # Run the multimodal agent for analysis
                                    response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])

                                # Display the result
                                st.subheader("Analysis Result")
                                st.markdown(response.content)

                            except Exception as error:
                                st.error(f"An error occurred during analysis: {error}")
                            finally:
                                Path(video_path).unlink(missing_ok=True)
elif file_type == "Tool Interaction":
    st.sidebar.subheader("Available Tools")
    for tool_name in tools.keys():
        st.sidebar.write(f"- {tool_name}")

    user_query_tool = st.text_input("Ask anything for the tools to handle")
    if st.button("Submit"):
        with st.spinner("Processing your query"):
            result = tool_agent.invoke({"input": user_query_tool})
            st.write("Response:", result["output"])