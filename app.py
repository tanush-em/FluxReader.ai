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
from gtts import gTTS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Utility Functions
def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return os.path.basename(max(paths, key=os.path.getctime))

def text_to_audio(text):
    tts = gTTS(text=text, lang='en', slow=False)
    mp3_file = "temp_audio.mp3"
    tts.save(mp3_file)
    return mp3_file

def save_uploaded_file(uploaded_file, directory):
    try:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return st.success(f"Saved file: {uploaded_file.name} to {directory}")
    except Exception as e:
        return st.error(f"Error saving file: {e}")

# PDF Extraction Function with Multi-Column Support
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

# PDF Reading and Tracking Functions
def track_pdf_reading(pages, current_page=0):
    st.session_state.current_page = current_page
    st.session_state.total_pages = len(pages)
    
    page_content = pages[current_page]
    st.text_area("Current Page Content", page_content, height=300)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous Page") and st.session_state.current_page > 0:
            st.session_state.current_page -= 1
            track_pdf_reading(pages, st.session_state.current_page)
    
    with col2:
        if st.button("Next Page") and st.session_state.current_page < st.session_state.total_pages - 1:
            st.session_state.current_page += 1
            track_pdf_reading(pages, st.session_state.current_page)
    
    st.write(f"Page {st.session_state.current_page + 1} of {st.session_state.total_pages}")

def pdf_reader_with_pause(pages):
    if 'reading_paused' not in st.session_state:
        st.session_state.reading_paused = False
    
    if not st.session_state.reading_paused:
        track_pdf_reading(pages)
    
    pause_col, resume_col = st.columns(2)
    with pause_col:
        if st.button("Pause Reading"):
            st.session_state.reading_paused = True
            st.write("Reading paused. Use resume to continue.")
    
    with resume_col:
        if st.button("Resume Reading"):
            st.session_state.reading_paused = False
            track_pdf_reading(pages, st.session_state.current_page)

# LLM and Embedding Configuration
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.3,
    max_tokens=1000,
)

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings":False}
)

def text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=20,
        length_function=len,
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

groq_client = Groq()

def transcribe_audio(filename):
    try:
        with open(filename, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(filename, file.read()), 
                model="distil-whisper-large-v3-en", 
                prompt="Specify context or spelling",  
                response_format="json",  
                language="en", 
                temperature=0.0  
            )
            return transcription.text
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return ""

# Streamlit App Configuration
st.set_page_config(
    page_title="FluxReader.ai",
    page_icon="📚",  
    layout="wide"
)

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

    st.image("image.png", caption="Audio Powered RAG", width=400)
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

    # Modify this part of the PDF processing code
    if uploaded_file is not None:
        # Save uploaded file
        save_uploaded_file(uploaded_file, upload_dir)
        file_name = uploaded_file.name
        file_path = f"uploaded_files/{file_name}"
        
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
        persist_directory = f"chroma_storage_{file_name.split('.')[0]}"
        
        try:
            # Initialize ChromaDB Client
            client = chromadb.PersistentClient(path=persist_directory)
            
            # Create or load vectorstore
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=persist_directory,
                collection_name=file_name.split(".")[0]
            )
            
            # Add documents if vectorstore is empty
            if len(vectorstore.get()['documents']) == 0:
                vectorstore.add_documents(documents)  # Pass Document objects here
                st.success(f"Loaded {len(pages)} pages into vectorstore")
        
        except Exception as e:
            st.error(f"Error setting up vectorstore: {e}")
            vectorstore = None


        if vectorstore and st.session_state.get("start_process"):
            st.header("PDF Reader")
            pdf_reader_with_pause(pages)
        
        st.title("Audio Recorder - Ask Question")
        audio = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop recording",
            just_once=False,
            key='recorder'
        )

        if audio:
            st.audio(audio['bytes'], format='audio/wav')
            with open("recorded_audio.wav", "wb") as f:
                f.write(audio['bytes'])
            
            with st.spinner("Transcribing Audio..."):
                transcription = transcribe_audio("recorded_audio.wav")
                st.write("Transcription:", transcription)

            if transcription and vectorstore:
                with st.spinner("Generating Response..."):
                    response = answer_question(transcription, vectorstore)
                    audio_response = text_to_audio(response)
                    
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    st.session_state.chat_history.append({
                        "question": transcription, 
                        "response": response
                    })
                    
                    message(transcription, is_user=True)
                    message(response, is_user=False)
                    st.audio(audio_response, format='audio/mp3')
