import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from PyPDF2 import PdfReader

load_dotenv()

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# app config (moved to the top)
st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖")
st.title("GeniusBot: AI-Powered Assistance with PDF Insight")

# File uploader
pdf_file = st.file_uploader(label="Upload PDF", type=["pdf"], key="pdf_uploader", help="You can upload a PDF to ask questions based on it.", label_visibility="hidden")

# Flag to track whether a new PDF file has been uploaded
new_pdf_uploaded = False

# Store the content of the PDF file
pdf_text = None

# OpenAI client setup
client = OpenAI(
    api_key="22edd11be66914ad65cf9ca6010c2ded8aba2cb8e057068e5bbc579a83a07208",
    base_url="https://llm.mdb.ai"
)

def get_response(user_query, chat_history):
    global pdf_text
    template = """ You are a helpful assistant. Chat history: {chat_history} User question: {user_question} """
    prompt = ChatPromptTemplate.from_template(template)

    if pdf_text and "pdf" in user_query.lower():
        user_question = "Based on the uploaded PDF, here is the extracted text:\\n" + pdf_text
        chat_history_with_pdf = chat_history + [HumanMessage(content=user_question)]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_query},
            {"role": "system", "content": "You have uploaded a PDF. Here is the extracted text:"},
            {"role": "user", "content": pdf_text}
        ]
    else:
        chat_history_with_pdf = chat_history
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_query}
        ]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=False
    )
    return completion.choices[0].message.content

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am a GeniusBot: AI-Powered Assistance with PDF Insight. Developed by Vinayak Singh using gpt-3.5-turbo and Streamlit. How can I help you?")]

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    if pdf_file is not None and not new_pdf_uploaded:
        with st.spinner("Uploading PDF..."):
            pdf_text = extract_text_from_pdf(pdf_file)
            new_pdf_uploaded = True

    response = get_response(user_query, st.session_state.chat_history)
    st.session_state.chat_history.append(AIMessage(content=response))
    with st.chat_message("AI"):
        st.write(response)


# Adjust CSS styles for block-container
st.markdown(
    """
    <style>
    @media (max-width:576px) {
    .block-container.st-emotion-cache-11kmii3.ea3mdgi5 {
    max-width: 100%;
    .block-container.st-emotion-cache-139wi93{
    max-width: max-content;
    }
    }
}
    </style>
    """,
    unsafe_allow_html=True,
)