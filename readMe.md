## README: **FluxReader.ai**  

### Introduction
**FluxReader.ai** is an advanced RAG (Retrieval-Augmented Generation) QA system that lets users upload PDFs and ask questions about their content using voice input. The system combines **GROQ AI**, **ChromaDB**, and **LangChain** to transcribe audio, retrieve relevant information, and provide text and audio-based responses.

---

### Features
1. **PDF Chat**:
   - Upload PDFs and interact with them conversationally.
   - Utilizes GROQ AI and vector databases for efficient document retrieval.

2. **Voice Interaction**:
   - Record questions using your microphone.
   - Automatic transcription and response synthesis using text-to-speech.

3. **Persistence**:
   - Retains previously processed documents in a persistent ChromaDB store.

4. **Interactive UI**:
   - User-friendly interface with **Streamlit**.

---

### Prerequisites
- **Python**: Version 3.8 or higher.
- **Libraries**:
  - `streamlit`
  - `streamlit-mic-recorder`
  - `streamlit-chat`
  - `langchain`
  - `chromadb`
  - `pydub`
  - `gTTS`
  - `dotenv`
  - `PyPDFLoader`
  - `HuggingFaceBgeEmbeddings`

---

### Installation
Follow these steps to set up the project locally:

#### Step 1: Clone the Repository
```bash
git clone https://github.com/your-repo/fluxreader-ai.git
cd fluxreader-ai
```

#### Step 2: Create a Virtual Environment
```bash
python -m venv env
source env/bin/activate   # On Windows, use `env\Scripts\activate`
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Set Up Environment Variables
1. Create a `.env` file in the root directory.
2. Add the following keys:
   ```
   GROQ_API_KEY=<Your_GROQ_API_Key>
   ```

---

### Usage
1. **Start the Application**
   ```bash
   streamlit run app.py
   ```
   - The app will be accessible at `http://localhost:8501`.

2. **Upload PDF**:
   - Select and upload a PDF file.
   - Ensure the filename has no spaces.

3. **Record Audio**:
   - Use the microphone recorder to ask questions about the uploaded PDF.

4. **View and Hear Responses**:
   - See the transcribed question and text-based response.
   - Listen to the audio synthesis of the response.

---

### Folder Structure
```
fluxreader-ai/
│
├── app.py                  # Main Streamlit app script
├── requirements.txt        # List of dependencies
├── uploaded_files/         # Folder for uploaded PDFs
├── chromanew/              # Persistent vector database directory
├── image.png               # UI banner image
├── .env                    # Environment variables
└── README.md               # Documentation
```

---

### Key Functions
- **`save_uploaded_file`**: Saves uploaded PDFs to a directory.
- **`text_splitter`**: Splits PDF content into manageable chunks.
- **`text_to_audio`**: Converts text responses to audio using `gTTS`.
- **`transcribe_audio`**: Transcribes recorded audio using GROQ.

---

### Troubleshooting
1. **Common Errors**:
   - *FileNotFoundError*: Ensure the `uploaded_files/` directory exists.
   - *API Error*: Verify your GROQ API key in the `.env` file.

2. **Check Logs**:
   - Streamlit logs can provide detailed error messages.

---

### License
MIT License. See `LICENSE` for details.

---

### Future Enhancements
1. Rigorous state management in 'session.state'
2. Enhanced UI for multi-document queries.
3. Integration with additional LLMs and vector stores.
4. Support for multiple languages.