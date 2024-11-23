import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# parse a list of pdf and extract the content of all, concatenate them and return the entire corpus
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text): 
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks 

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks,
                                   embedding = embeddings)
    return vectorstore

def main():
    load_dotenv()
    st.set_page_config(
        page_title = "FluxReader.ai",
        page_icon = ":microphone:"
    )
    st.header("FluxReader.ai")
    st.text_input("Ask question about your document...")

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
                        "Upload your PDFs here", 
                         accept_multiple_files = True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # parse text
                raw_text = get_pdf_text(pdf_docs)
                # get text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # create vextor store



if __name__ == '__main__':
    main()