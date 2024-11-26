import streamlit as st
from time import sleep
from streamlit_mic_recorder import mic_recorder
from streamlit_chat import message
import os
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
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

# Configuration
upload_dir = "uploaded_files"
os.makedirs(upload_dir, exist_ok=True)

# LLM and Embedding Configuration
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.3,
    max_tokens=1000,
)

# Embeddings Configuration
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

# Groq Client for Transcription
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
        <h1 style='text-align: center;'>
            🎧 Talk with your PDFs 📚 
        </h1>
        <h5 style='text-align: center;'>
            VoiceIO RAG based QA system
        </h5>
        """,
        unsafe_allow_html=True
    )

    st.image("image.png", caption="Audio Powered RAG", use_container_width=True)
    
    if st.button("Stop Process"):
        st.session_state.stop = True 

    if hasattr(st.session_state, 'stop') and st.session_state.stop:
        st.write("The process has been stopped. You can refresh the page to restart.")

with col2:
    st.markdown("<h1 style='text-align: center; color: white;'>FluxReader.ai</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: grey;'>Converse with PDFs using GROQ</h2>", unsafe_allow_html=True)
    
    # PDF Upload Section
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Vectorstore and Processing
    if uploaded_file is not None:
        # Save uploaded file
        save_uploaded_file(uploaded_file, upload_dir)
        file_name = uploaded_file.name
        
        # Load PDF
        loader = PyPDFLoader(f"uploaded_files/{file_name}")
        pages = loader.load_and_split(text_splitter())
        
        # Create unique persist directory
        persist_directory = f"chroma_storage_{file_name.split('.')[0]}"
        
        try:
            # Initialize ChromaDB Client with simplified approach
            client = chromadb.PersistentClient(path=persist_directory)
            
            # Create or load vectorstore
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=persist_directory,
                collection_name=file_name.split(".")[0]
            )
            
            # Add documents if vectorstore is empty
            if len(vectorstore.get()['documents']) == 0:
                vectorstore.add_documents(pages)
                st.success(f"Loaded {len(pages)} pages into vectorstore")
            
        except Exception as e:
            st.error(f"Error setting up vectorstore: {e}")
            vectorstore = None

    # Process Initialization
    if 'start_process' not in st.session_state:
        st.session_state.start_process = False

    if st.button("Start Process"):
        st.session_state.start_process = True

    # Main Processing Section
    if st.session_state.start_process and uploaded_file:
        # File Selection
        options = os.listdir("uploaded_files")
        file_name = st.selectbox("Select a file:", options)
        
        # Audio Recording
        st.title("Audio Recorder - Ask Question")
        audio = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop recording",
            just_once=False,
            key='recorder'
        )

        if audio:
            # Save and process audio
            st.audio(audio['bytes'], format='audio/wav')
            with open("recorded_audio.wav", "wb") as f:
                f.write(audio['bytes'])
            
            # Transcribe Audio
            with st.spinner("Transcribing Audio..."):
                transcription = transcribe_audio("recorded_audio.wav")
                st.write("Transcription:", transcription)

            # Generate Response
            if transcription and vectorstore:
                with st.spinner("Generating Response..."):
                    response = answer_question(transcription, vectorstore)
                    
                    # Text-to-Speech
                    audio_response = text_to_audio(response)
                    
                    # Display Chat History
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    st.session_state.chat_history.append({
                        "question": transcription, 
                        "response": response
                    })
                    
                    # Display Messages
                    message(transcription, is_user=True)
                    message(response, is_user=False)
                    
                    # Audio Playback
                    st.audio(audio_response, format='audio/mp3')
