import streamlit as st
from time import sleep
from streamlit_mic_recorder import mic_recorder
from streamlit_chat import message
import os
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import pygame
import shutil
import atexit
from pathlib import Path

# Import utility functions
from utils import (
    cleanup_session_files,
    ensure_audio_file_exists,
    text_to_audio,
    voice_read_pdf,
    extract_pdf_text,
    save_uploaded_file,
    cleanup_temp_directories
)

# Load environment variables from .env file
load_dotenv()

# Directory setup
BASE_TEMP_DIR = "temp"
UPLOAD_DIR = os.path.join(BASE_TEMP_DIR, "uploaded_files")
AUDIO_DIR = os.path.join(BASE_TEMP_DIR, "audio")
CHROMADB_DIR = os.path.join(BASE_TEMP_DIR, "chromadb")
os.makedirs(AUDIO_DIR, exist_ok=True)  # Create audio directory if it doesn't exist

# Register cleanup function to delete temporary audio files on exit
atexit.register(lambda: cleanup_session_files(AUDIO_DIR))

# Initialize language model
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.5,
    max_tokens=1000,
)

# Initialize embeddings
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

# Answer user questions using the vectorstore and LLM
def answer_question(question, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa.invoke({"query": question})
    return result['result']

# Configure Streamlit app
st.set_page_config(
    page_title="FluxReader.ai",
    page_icon="📚",  
    layout="wide"
)

def main():
    # Layout: two columns
    col1, col2 = st.columns([1, 2])

    # Sidebar column (col1)
    with col1:
        st.markdown(
            """
            <h1 style='text-align: center;'>🎧 Talk with your PDFs 📚</h1>
            <h5 style='text-align: center;'>VoiceIO RAG-based QA system</h5>
            """,
            unsafe_allow_html=True
        )

        st.image("assets/image.png", caption="Audio Powered RAG", use_container_width=True)

        # Add custom button styling
        st.markdown("""
            <style>
                .stButton > button {
                    background-color: white;
                    color: black;
                    border: none;
                    padding: 0.5rem 1rem;
                    border-radius: 0.3rem;
                    font-weight: bold;
                    font-size: 1rem;
                    transition: background-color 0.3s ease, transform 0.2s ease;
                }
                .stButton > button:hover {
                    background-color: rgb(235, 35, 16);
                    color: black;
                    transform: scale(1.05);
                }
                div[data-testid="stButton"] {
                    margin-top: 1rem;
                    text-align: center;
                }
            </style>
        """, unsafe_allow_html=True)

        # Stop process button
        if st.button("Stop Process"):
            # Set session state to stop
            st.session_state.stop = True
            
            # Clean up temporary directories
            cleanup_temp_directories([UPLOAD_DIR, AUDIO_DIR, CHROMADB_DIR])
            
            st.success("Temporary files have been deleted. Refresh to restart.")

        if st.session_state.get('stop', False):
            st.write("The process has been stopped. Refresh to restart.")

    # Main content column (col2)
    with col2:
        st.markdown("<h1 style='text-align: center;'>FluxReader.ai</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>Converse with PDFs using GROQ</h2>", unsafe_allow_html=True)
        st.write("")

        # Upload PDF
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        st.write("WARNING: Ensure the uploaded PDF has no blank spaces in the text.")

        if uploaded_file is not None:
            # Save the uploaded file
            save_uploaded_file(uploaded_file, UPLOAD_DIR)
            file_name = uploaded_file.name
            file_path = os.path.join(UPLOAD_DIR, file_name)

            # Extract text from the PDF
            full_text = extract_pdf_text(file_path)

            # Split text into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            pages = text_splitter.split_text(full_text)

            # Convert each chunk into a Document object
            documents = [Document(page_content=page, metadata={"source": file_name}) for page in pages]

            # Create or load vectorstore
            persist_directory = os.path.join(CHROMADB_DIR, f"storage_{file_name.split('.')[0]}")
            try:
                client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=Settings(anonymized_telemetry=False, is_persistent=True)
                )
                vectorstore = Chroma(client=client, embedding_function=embeddings, collection_name=file_name.split(".")[0])
                if len(vectorstore.get()['documents']) == 0:
                    vectorstore.add_documents(documents)
                    st.success(f"Loaded {len(pages)} pages into vectorstore")
            except Exception as e:
                st.error(f"Error setting up vectorstore: {e}")
                vectorstore = None

            # PDF voice reader
            st.header("PDF Voice Reader")
            voice_read_pdf(pages, AUDIO_DIR)

            # Audio recorder for questions
            st.title("Audio Recorder - Ask Question")
            audio = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="⏹️ Stop Recording", just_once=False)

            if audio:
                try:
                    audio_file = ensure_audio_file_exists(AUDIO_DIR, "recorded_audio.wav")
                    with open(audio_file, "wb") as f:
                        f.write(audio['bytes'])
                    st.audio(audio['bytes'], format='audio/wav')

                    # Transcription and question answering
                    with st.spinner("Transcribing Audio..."):
                        try:
                            groq_client = Groq()
                            with open(audio_file, "rb") as file:
                                transcription = groq_client.audio.transcriptions.create(
                                    file=(file.name, file.read()), 
                                    model="distil-whisper-large-v3-en",  
                                    response_format="json",  
                                    language="en", 
                                )
                                transcription_text = transcription.text
                            st.write("Transcription:", transcription_text)

                            if transcription_text and vectorstore:
                                with st.spinner("Generating Response..."):
                                    response = answer_question(transcription_text, vectorstore)
                                    audio_response = text_to_audio(response, AUDIO_DIR, "response_audio.mp3")

                                    if 'chat_history' not in st.session_state:
                                        st.session_state.chat_history = []

                                    st.session_state.chat_history.append({"question": transcription_text, "response": response})
                                    message(transcription_text, is_user=True)
                                    message(response, is_user=False)
                                    st.audio(audio_response, format='audio/mp3')
                        except Exception as e:
                            st.error(f"Transcription Error: {e}")
                except Exception as e:
                    st.error(f"Error processing audio: {e}")

if __name__ == "__main__":
    main()
