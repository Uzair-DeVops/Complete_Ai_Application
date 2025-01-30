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





import requests
from PIL import Image
from io import BytesIO

# Configure API Key


image_model = genai.GenerativeModel("gemini-2.0-flash-exp")





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
# st.title("Tool_calling, Rag, video Summarizer Application Powered BY Gemini") 
st.sidebar.title("Options")

# Sidebar: File Upload or Tool Selection
file_type = st.sidebar.radio("Choose an option", ["Image","Video","PDF",  "Tool Interaction" ,"Translator"])

if file_type == "PDF":
    st.title("PDF Summarizer") 
    st.write("upload your Pdf")

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
    st.title("Video Summarizer")
    st.write("upload your video")
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


elif file_type == "Image":
        # Streamlit UI
        st.title("Image Description Generator")

        # Sidebar for file uploader
        with st.sidebar:
            uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                st.sidebar.success("Image uploaded! Now you can summarize it. Press the 'Generate Description' button.")

        # Input for image URL
        url = st.text_input("Enter Image URL: or Upload The Image" )

        # Display image as soon as uploaded or URL is pasted
        def display_image():
            if uploaded_file is not None:
                return Image.open(uploaded_file)
            elif url:
                response = requests.get(url)
                return Image.open(BytesIO(response.content))
            return None

        img = display_image()
        if img:
            st.image(img, caption="Uploaded Image", use_container_width=True)
            st.write("Press the 'Generate Description' button to summarize the image.")

        if st.button("Generate Description"):
            try:
                if img:
                    # Generate description
                    prompt = "Describe the image in detail"
                    response = image_model.generate_content([prompt, img])
                    
                    # Display the response
                    st.subheader("Generated Description:")
                    st.write(response.text)
            except Exception as e:
                st.error(f"Error: {e}")

            

elif file_type == "Translator":
    
# App Title
        # st.set_page_config(page_title="Multilingual Translator", layout="wide" ,)
        st.title("🌐 Multilingual Translator")
        st.write("Translate your ✍️ text or  📄 PDF content into multiple languages with ease!")

        # Sidebar
        st.sidebar.header("📄 PDF Translator")
        uploaded_file = st.sidebar.file_uploader("Upload your PDF here:", type="pdf")

        # Language Selection
        language = st.sidebar.selectbox(
            "🌍 Select a language for translation:",
            [
                "Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian", "Azerbaijani", "Basque", "Belarusian", 
            "Bengali", "Bosnian", "Bulgarian", "Catalan", "Cebuano", "Chinese", "Corsican", "Croatian", 
            "Czech", "Danish", "Dutch", "English", "Esperanto", "Estonian", "Filipino", "Finnish", "French", 
            "Frisian", "Galician", "Georgian", "German", "Greek", "Gujarati", "Haitian Creole", "Hausa", 
            "Hawaiian", "Hebrew", "Hindi", "Hmong", "Hungarian", "Icelandic", "Igbo", "Indonesian", "Irish", 
            "Italian", "Japanese", "Javanese", "Kannada", "Kazakh", "Khmer", "Kinyarwanda", "Korean", 
            "Kurdish", "Kyrgyz", "Lao", "Latin", "Latvian", "Lithuanian", "Luxembourgish", "Macedonian", 
            "Malagasy", "Malay", "Malayalam", "Maltese", "Maori", "Marathi", "Mongolian", "Myanmar (Burmese)", 
            "Nepali", "Norwegian", "Nyanja (Chichewa)", "Odia (Oriya)", "Pashto", "Persian", "Polish", 
            "Portuguese", "Punjabi", "Romanian", "Russian", "Samoan", "Scots Gaelic", "Serbian", "Sesotho", 
            "Shona", "Sindhi", "Sinhala (Sinhalese)", "Slovak", "Slovenian", "Somali", "Spanish", "Sundanese", 
            "Swahili", "Swedish", "Tajik", "Tamil", "Tatar", "Telugu", "Thai", "Turkish", "Turkmen", "Ukrainian", 
            "Urdu", "Uyghur", "Uzbek", "Vietnamese", "Welsh", "Xhosa", "Yiddish", "Yoruba", "Zulu"
            ]
        )

        # PDF Translation
        if uploaded_file:
            pdf_reader = PdfReader(uploaded_file)
            pdf_text = "".join([page.extract_text() for page in pdf_reader.pages])

            st.sidebar.subheader("📝 Extracted PDF Content")
            st.sidebar.text_area("Preview PDF Content:", pdf_text[:500], height=150)

            if st.sidebar.button("Translate PDF"):
                if pdf_text.strip():
                    prompt = f"""
                    Act as a professional translator. Translate the following PDF content into **{language}**.
                    Preserve the tone and structure of the original text.

                    **PDF Content:**  
                    {pdf_text}
                    """
                    response = llm.invoke(prompt)
                    translated_text = response.content

                    st.subheader("📖 Translated PDF Content")
                    st.text_area("Translation:", translated_text, height=250)

                    translated_file_path = "translated_file.txt"
                    with open(translated_file_path, "w", encoding="utf-8") as f:
                        f.write(translated_text)
                    with open(translated_file_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Translated PDF",
                            data=f,
                            file_name="translated_file.txt",
                            mime="text/plain",
                        )
                    os.remove(translated_file_path)
                else:
                    st.sidebar.error("No content found in the uploaded PDF.")

        # Text Translation
        user_text = st.text_area("Enter your text:", placeholder="Type your text here...")

        if st.button("Translate Text"):
            if user_text.strip():
                prompt = f"""
                Act as a professional translator. Translate the following text into **{language}**.
                Preserve the tone and structure of the original text.

                **Text Content:**  
                {user_text}
                """
                response = llm.invoke(prompt)
                translated_text = response.content

                st.subheader("📝 Translated Text")
                st.text_area("Translation:", translated_text, height=250)

                translated_file_path = "translated_text.txt"
                with open(translated_file_path, "w", encoding="utf-8") as f:
                    f.write(translated_text)
                with open(translated_file_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Translated Text",
                        data=f,
                        file_name="translated_text.txt",
                        mime="text/plain",
                    )
                os.remove(translated_file_path)
            else:
                st.error("Please enter some text for translation.")









elif file_type == "Tool Interaction":
    st.title("🤖 Tool Interaction")
    st.sidebar.subheader("Available Tools")
    for tool_name in tools.keys():
        st.sidebar.write(f"- {tool_name}")

    user_query_tool = st.text_input("Ask anything for the tools to handle")
    if st.button("Submit"):
        with st.spinner("Processing your query"):
            result = tool_agent.invoke({"input": user_query_tool})
            st.write("Response:", result["output"])