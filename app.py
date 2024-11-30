# Dependancies and Packages
import streamlit as st
from time import sleep
from streamlit_mic_recorder import mic_recorder
from streamlit_chat import message
import os
import fitz  # PyMuPDF
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA
import chromadb
from chromadb.config import Settings
from gtts import gTTS
from dotenv import load_dotenv
import pygame
import shutil
import atexit
from pathlib import Path

# Load environment variables
load_dotenv()

BASE_TEMP_DIR = "temp"
UPLOAD_DIR = os.path.join(BASE_TEMP_DIR, "uploaded_files")
AUDIO_DIR = os.path.join(BASE_TEMP_DIR, "audio")
CHROMADB_DIR = os.path.join(BASE_TEMP_DIR, "chromadb")

os.makedirs(AUDIO_DIR, exist_ok=True)

# Utility Functions

def ensure_audio_file_exists(filename):
    """Create an empty audio file if it doesn't exist"""
    filepath = os.path.join(AUDIO_DIR, filename)
    Path(filepath).touch(exist_ok=True)
    return filepath

def cleanup_session_files():
    """Cleanup temporary files at session end"""
    temp_files = [
        os.path.join(AUDIO_DIR, 'recorded_audio.wav'),
        os.path.join(AUDIO_DIR, 'temp_audio.mp3')
    ]
    for file in temp_files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except Exception as e:
            print(f"Error cleaning up {file}: {e}")

# Register cleanup function to run at exit
atexit.register(cleanup_session_files)

def text_to_audio(text, filename):
    filepath = os.path.join(AUDIO_DIR, filename)
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filepath)
        return filepath
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None
    
def cleanup_temp_files(pattern="page_*.mp3"):
    """Clean up temporary audio files"""
    import glob
    for file in glob.glob(os.path.join(AUDIO_DIR, pattern)):
        os.remove(file)

def voice_read_pdf(pages, start_page=0):
    """Voice read PDF pages with play/pause/resume functionality"""
    # Initialize pygame mixer if not already initialized
    if not pygame.mixer.get_init():
        pygame.mixer.init()

    # Initialize reading state if not exists
    if 'reading_state' not in st.session_state:
        st.session_state.reading_state = {
            'current_page': start_page,
            'is_playing': False,
            'audio_files': []
        }

    # Prepare audio files for pages if not already done
    if not st.session_state.reading_state['audio_files']:
        cleanup_temp_files()  # Clean any previous temp files
        for idx, page in enumerate(pages[start_page:], start=start_page):
            audio_file = text_to_audio(page, f"page_{idx}.mp3")
            st.session_state.reading_state['audio_files'].append(audio_file)

    # Get current page from reading state
    current_page = st.session_state.reading_state['current_page']
    
    # Display current page content
    st.text_area("Current Page Content", pages[current_page], height=300)

    # Audio control columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Previous Page") and current_page > 0:
            pygame.mixer.music.stop()
            st.session_state.reading_state['current_page'] -= 1
            st.rerun()

    with col2:
        if st.button("Next Page") and current_page < len(pages) - 1:
            pygame.mixer.music.stop()
            st.session_state.reading_state['current_page'] += 1
            st.rerun()

    with col3:
        if st.button("Play"):
            if not st.session_state.reading_state['is_playing']:
                pygame.mixer.music.load(st.session_state.reading_state['audio_files'][current_page])
                pygame.mixer.music.play()
                st.session_state.reading_state['is_playing'] = True

    with col4:
        if st.button("Pause/Resume"):
            if st.session_state.reading_state['is_playing']:
                pygame.mixer.music.pause()
                st.session_state.reading_state['is_playing'] = False
            else:
                pygame.mixer.music.unpause()
                st.session_state.reading_state['is_playing'] = True

    st.write(f"Page {current_page + 1} of {len(pages)}")

def extract_pdf_text(pdf_path, footer_margin=50, header_margin=50):
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        page_text = ""
        sorted_blocks = sorted(blocks, key=lambda b: b['bbox'][1])
        
        for block in sorted_blocks:
            if block['type'] == 0:  # Text block
                bbox = block['bbox']
                if (bbox[1] >= header_margin and bbox[3] <= page.rect.height - footer_margin):
                    for line in block['lines']:
                        page_text += ' '.join([span['text'] for span in line['spans']]) + ' '
        
        full_text += page_text + "\n\n"
    
    return full_text

def save_uploaded_file(uploaded_file, directory):
    try:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return st.success(f"Saved file: {uploaded_file.name} to {directory}")
    except Exception as e:
        return st.error(f"Error saving file: {e}")

# LLM Configuration and other existing functions remain the same
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

        m = st.markdown(""" <style> 
                        div.stButton > button:first-child 
                        { background-color: rgb(204, 49, 49); } 
                        </style>""", 
                        unsafe_allow_html=True) 
        b = st.button("test")
        if st.button("Stop Process"):
            st.session_state.stop = True 
        if hasattr(st.session_state, 'stop') and st.session_state.stop:
            st.write("The process has been stopped. You can refresh the page to restart.")

    with col2:
        st.markdown("<h1 style='text-align: center; color: white;'>FluxReader.ai</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: grey;'>Converse with PDFs using GROQ</h2>", unsafe_allow_html=True)
        
        upload_dir = "uploaded_files"
        os.makedirs(upload_dir, exist_ok=True)
        
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
            voice_read_pdf(pages)
            
            st.title("Audio Recorder - Ask Question")
            audio = mic_recorder(
                start_prompt="Start recording",
                stop_prompt="Stop recording",
                just_once=False,
                key='recorder'
            )

            if audio:
                try:
                    audio_file = ensure_audio_file_exists("recorded_audio.wav")
                    with open(audio_file, "wb") as f:
                        f.write(audio['bytes'])
                    st.audio(audio['bytes'], format='audio/wav')
                
                    with st.spinner("Transcribing Audio..."):
                        try:
                            groq_client = Groq()
                            # Use the audio_file variable instead of direct filename
                            with open(audio_file, "rb") as file:
                                transcription = groq_client.audio.transcriptions.create(
                                    file=(file.name, file.read()), 
                                    model="distil-whisper-large-v3-en", 
                                    prompt="Specify context or spelling",  
                                    response_format="json",  
                                    language="en", 
                                    temperature=0.0  
                                )
                                transcription_text = transcription.text
                            
                            st.write("Transcription:", transcription_text)

                            if transcription_text and vectorstore:
                                with st.spinner("Generating Response..."):
                                    response = answer_question(transcription_text, vectorstore)
                                    audio_response = text_to_audio(response, "response_audio.mp3")
                                    
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