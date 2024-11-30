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
    ensure_audio_file_exists,
    cleanup_session_files,
    text_to_audio,
    cleanup_temp_files,
    voice_read_pdf,
    extract_pdf_text,
    save_uploaded_file
)

# Load environment variables
load_dotenv()

# Directory Setup
BASE_TEMP_DIR = "temp"
UPLOAD_DIR = os.path.join(BASE_TEMP_DIR, "uploaded_files")
AUDIO_DIR = os.path.join(BASE_TEMP_DIR, "audio")
CHROMADB_DIR = os.path.join(BASE_TEMP_DIR, "chromadb")

os.makedirs(AUDIO_DIR, exist_ok=True)

# Register cleanup function to run at exit
atexit.register(lambda: cleanup_session_files(AUDIO_DIR))

# LLM Configuration
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.5,
    max_tokens=1000,
)

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings":False}
)

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

# Streamlit App Configuration
st.set_page_config(
    page_title="FluxReader.ai",
    page_icon="📚",  
    layout="wide"
)

def main():
    # Main App Layout
    col1, col2 = st.columns([1,2])  

    with col1:
        st.markdown(
            """
            <h1 style='text-align: center;'>🎧 Talk with your PDFs 📚</h1>
            <h5 style='text-align: center;'>VoiceIO RAG based QA system</h5>
            """,
            unsafe_allow_html=True
        )

        st.image("image.png", caption="Audio Powered RAG", use_container_width=True)

        # Add the unified custom CSS styling
        st.markdown("""
            <style>
                .stButton > button,
                div[data-testid="stButton"] button[kind="primary"] {
                    background-color: white;
                    color: black;
                    border: none;
                    padding: 0.5rem 1rem;
                    border-radius: 0.3rem;
                    font-weight: bold;
                    font-size: 1rem;
                    transition: background-color 0.3s ease, transform 0.2s ease;
                }

                .stButton > button:hover,
                div[data-testid="stButton"] button[kind="primary"]:hover {
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

        if st.button("Stop Process"):
            st.session_state.stop = True 
        if hasattr(st.session_state, 'stop') and st.session_state.stop:
            st.write("The process has been stopped. You can refresh the page to restart.")

    with col2:
        st.markdown("<h1 style='text-align: center; color: white;'>FluxReader.ai</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: grey;'>Converse with PDFs using GROQ</h2>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            # Save uploaded file
            save_uploaded_file(uploaded_file, UPLOAD_DIR)
            file_name = uploaded_file.name
            file_path = os.path.join(UPLOAD_DIR, file_name)
            
            # Extract text using PyMuPDF
            full_text = extract_pdf_text(file_path)
            
            # Split text into pages
            text_splitter_func = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
            )
            pages = text_splitter_func.split_text(full_text)
            
            # Convert pages to Document objects
            documents = [Document(page_content=page, metadata={"source": file_name}) for page in pages]
            
            # Create unique persist directory
            persist_directory = os.path.join(CHROMADB_DIR, f"storage_{file_name.split('.')[0]}")
            
            try:
                # Initialize ChromaDB Client
                client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        is_persistent=True
                    )
                )
                
                # Create or load vectorstore
                vectorstore = Chroma(
                    client=client,  
                    embedding_function=embeddings,
                    collection_name=file_name.split(".")[0]
                )
                
                # Add documents if vectorstore is empty
                if len(vectorstore.get()['documents']) == 0:
                    vectorstore.add_documents(documents)
                    st.success(f"Loaded {len(pages)} pages into vectorstore")
            
            except Exception as e:
                st.error(f"Error setting up vectorstore: {e}")
                vectorstore = None

            st.header("PDF Voice Reader")
            voice_read_pdf(pages, AUDIO_DIR)
            
            st.title("Audio Recorder - Ask Question")
            
            audio = mic_recorder(
                start_prompt="🎙️ Start Recording",
                stop_prompt="⏹️ Stop Recording",
                just_once=False,
                key="recorder"
            )

            if audio:
                try:
                    audio_file = ensure_audio_file_exists(AUDIO_DIR, "recorded_audio.wav")
                    with open(audio_file, "wb") as f:
                        f.write(audio['bytes'])
                    st.audio(audio['bytes'], format='audio/wav')
                
                    with st.spinner("Transcribing Audio..."):
                        try:
                            groq_client = Groq()
                            with open(audio_file, "rb") as file:
                                transcription = groq_client.audio.transcriptions.create(
                                    file=(file.name, file.read()), 
                                    model="distil-whisper-large-v3-en",    prompt="Specify context or spelling",  
                                    response_format="json",  
                                    language="en", 
                                    temperature=0.0  
                                )
                                transcription_text = transcription.text
                            
                            st.write("Transcription:", transcription_text)

                            if transcription_text and vectorstore:
                                with st.spinner("Generating Response..."):
                                    response = answer_question(transcription_text, vectorstore)
                                    audio_response = text_to_audio(response, AUDIO_DIR, "response_audio.mp3")
                                    
                                    if 'chat_history' not in st.session_state:
                                        st.session_state.chat_history = []
                                    
                                    st.session_state.chat_history.append({
                                        "question": transcription_text, 
                                        "response": response
                                    })
                                    
                                    message(transcription_text, is_user=True)
                                    message(response, is_user=False)
                                    st.audio(audio_response, format='audio/mp3')
                        except Exception as e:
                            st.error(f"Transcription Error: {e}")
                except Exception as e:
                    st.error(f"Error processing audio: {e}")

if __name__ == "__main__":
    main()
