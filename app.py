import streamlit as st
from time import sleep
#from st_audiorec import st_audiorec
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
from chromadb.config import Settings
import chromadb
from gtts import gTTS
from pydub import AudioSegment
import os
from dotenv import load_dotenv

load_dotenv()

chroma_setting = Settings(anonymized_telemetry=False)

def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    newest_file_path = max(paths, key=os.path.getctime)
    return os.path.basename(newest_file_path)

def text_to_audio(text):
    tts = gTTS(text=text, lang='en', slow=False)
    mp3_file = "temp_audio.mp3"
    tts.save(mp3_file)
    return mp3_file

def save_uploaded_file(uploaded_file, directory):
    try:
        with open(os.path.join(directory, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return st.success(f"Saved file: {uploaded_file.name} to {directory}")
    except Exception as e:
        return st.error(f"Error saving file: {e}")

upload_dir = "uploaded_files"
os.makedirs(upload_dir, exist_ok=True)

llm = ChatGroq(model_name="llama3-70b-8192",
    temperature=0.3,
    max_tokens=1000,
)

model_name ="BAAI/bge-small-en-v1.5"
model_kwargs ={"device":"cpu"}
encode_kwargs ={"normalize_embeddings":False}
embeddings = HuggingFaceBgeEmbeddings(model_name=model_name,
                                   model_kwargs=model_kwargs,
                                   encode_kwargs=encode_kwargs)

def text_splitter():
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=512,
      chunk_overlap=20,
      length_function=len,
  )
  return text_splitter

def answer_question(question,vectorstore):
  retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
  qa = RetrievalQA.from_chain_type(llm=llm,
                                   chain_type="stuff",
                                   retriever=retriever,
                                   return_source_documents=True)
  result = qa.invoke({"query": question})
  return result['result']

groq_client = Groq()

filename = "recorded_audio.wav"

def transcribe_audio(filename):
  with open(filename, "rb") as file:
      transcription = groq_client.audio.transcriptions.create(
        file=(filename, file.read()), 
        model="distil-whisper-large-v3-en", 
        prompt="Specify context or spelling",  
        response_format="json",  
        language="en", 
        temperature=0.0  
      )
      print(transcription.text)
  return transcription.text

if 'stop' not in st.session_state:
    st.session_state.stop = False

st.set_page_config(
    page_title="Audio and Book App",
    page_icon="📚",  
    layout="wide"
)

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

    st.image("image.png", caption="Audio Powered RAG",output_format="auto",use_container_width=True)
    
    if st.button("Stop Process"):
        st.session_state.stop = True 

    if st.session_state.stop:
        st.write("The process has been stopped. You can refresh the page to restart.")

with col2 :
    st.markdown("<h1 style='text-align: center; color: white;'>FluxReader.ai</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: grey;'>Converse with PDFs using GROQ</h2>", unsafe_allow_html=True)
    st.text("Upload your PDF to chat with it...")
    st.text("Make sure the name of the file doesnt have blank spaces.")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    persist_directory_path = "chromanew"
    if uploaded_file is not None:
        save_uploaded_file(uploaded_file, upload_dir)
        file_name = uploaded_file.name
        loader = PyPDFLoader(f"uploaded_files/{file_name}")
        pages = loader.load_and_split(text_splitter())
        persist_directory = persist_directory_path + "_" + file_name.split(".")[0]
        if os.path.exists(persist_directory):
            
            client = chromadb.PersistentClient(path=persist_directory, settings=chroma_setting)
            vectorstore = Chroma(embedding_function=embeddings,
                                 client = client,
                                persist_directory=persist_directory,
                                collection_name=file_name.split(".")[0],
                                client_settings=chroma_setting,
                                )
            print(f" The number of documents loaded in the vectorstore :{len(vectorstore.get()['documents'])}")
        else:
            client = chromadb.PersistentClient(path=persist_directory, settings=chroma_setting)
            vectorstore = Chroma(embedding_function=embeddings,
                                 client = client,
                                persist_directory=persist_directory,
                                collection_name=file_name.split(".")[0],
                                client_settings=chroma_setting
                                )
            MAX_BATCH_SIZE = 100

            for i in range(0,len(pages),MAX_BATCH_SIZE):
                i_end = min(len(pages),i+MAX_BATCH_SIZE)
                batch = pages[i:i_end]
    
                vectorstore.add_documents(batch)
    
            print(f" The number of documents loaded in the vectorstore :{len(vectorstore.get()['documents'])}")

    if 'start_process' not in st.session_state:
        st.session_state.start_process = False

    if st.button("Start Process"):
        st.session_state.start_process = True

    if st.session_state.start_process:
        options = os.listdir("uploaded_files")
        none_list = ["none"]
        options += none_list

        selected_option = st.selectbox("Select an option:", options)
        
        #
        if selected_option == "none":
            file_name = newest("uploaded_files")
        else:
            file_name = selected_option
        st.write(f"You selected: {selected_option}")
        st.title("Audio Recorder- Ask Question based on the selected option")
        with st.spinner("Audio Recording in progress..."):

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
                st.success("Audio Recording is completed!")
            
        with st.spinner("Transcribing Audio in progress ..."):
            text = transcribe_audio(filename)
            transcription = text
            st.markdown(text)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for i, chat in enumerate(st.session_state.chat_history):
            message(chat["question"], is_user=True, key=f"question_{i}")
            message(chat["response"], is_user=False, key=f"response_{i}")

        if transcription :
            with st.spinner("Syntesizing Response ....."):
                
                print(f"File_name :{file_name}")

                persist_directory_path = "chromanew"
                persist_directory = persist_directory_path + "_" + file_name.split(".")[0]
                client = chromadb.PersistentClient(path=persist_directory, settings=chroma_setting)
                vectorstore = Chroma(embedding_function=embeddings,
                                    client = client,
                                    persist_directory=persist_directory,
                                    collection_name=file_name.split(".")[0],
                                    client_settings=chroma_setting
                                    )
                response = answer_question(transcription,vectorstore)
                st.success("Response Generated")

            aud_file = text_to_audio(response)

            st.session_state.chat_history.append({"question": transcription, "response": response})

            message(transcription, is_user=True)
            message(response, is_user=False)
        
        st.text("Audio Playback")
        st.audio(aud_file, format='audio/wav', start_time=0)