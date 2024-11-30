
# FluxReader.ai

FluxReader.ai is an AI-powered application that lets you interact with PDF documents using voice and text. With advanced features like RAG-based QA, PDF voice reading, and audio transcription, FluxReader.ai transforms how you engage with documents.

---

## Features

1. **PDF Upload and Processing**:
   - Extracts text from PDF files, even with large documents.
   - Supports header and footer margin filtering to improve text extraction accuracy.

2. **Vector Search**:
   - Splits PDF content into chunks and stores them in a vectorstore for efficient retrieval.
   - Uses embeddings for semantic search to retrieve the most relevant answers to user queries.

3. **RAG-based QA System**:
   - Retrieves answers from the PDF content using a **Retrieval-Augmented Generation (RAG)** pipeline.
   - Powered by the Groq LLM and HuggingFace embeddings.

4. **Voice Interaction**:
   - Converts text to speech for reading PDFs aloud.
   - Records and transcribes user queries through an integrated microphone.
   - Generates and plays responses to queries.

5. **Streamlit-based Web Interface**:
   - A user-friendly dashboard for uploading, processing, and interacting with PDF files.
   - Sidebar for easy navigation and settings.
   - Chat-like interface to display user questions and AI responses.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (optional but recommended)

### Steps

1. Install python:
   Install Python if you dont have already. Python version >3.8 works fine.
   [Click Here](https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe) to install it.
   Make sure you check the "add to PATH" checkbox during installation.

2. Clone the repository:
   You can either use git CLI or by downloading the zip package of the repo.
   ```bash
   git clone https://github.com/yourusername/fluxreader-ai.git
   cd fluxreader-ai
   ```

3. Create a `.env` file to configure environment variables:
   ```env
   # .env file
   GROQ_API_KEY=<your_groq_api_key>
   ```

4. Run the batch file to setup the project locally:
   - Right click on the `run_app.bat` file and `Run as Administrator`.

---

## Usage

### Upload a PDF
1. Upload a PDF file through the provided file uploader.
2. Extracted text will be displayed, and the document will be processed into chunks.

### Voice-Read PDF
- Navigate between pages using "Previous" and "Next" buttons.
- Play or pause audio playback for each page's text.

### Ask Questions
1. Record a question using the **Audio Recorder** or type it in.
2. The AI will transcribe your question and provide an answer from the PDF content.

---

## File Structure

```
.
├── app.py               # Main Streamlit app
├── utils.py             # Utility functions
├── requirements.txt     # Dependencies
├── .env.example         # Environment variable example
├── temp/                # Temporary directory for audio, uploads, and vectorstore
└── README.md            # Project documentation
```

---

## Dependencies

- **langchain**: Language model orchestration
- **chromadb**: Vectorstore for embeddings
- **Groq**: LLM and transcription services
- **PyMuPDF**: PDF text extraction
- **Streamlit**: Web application framework
- **gTTS**: Google Text-to-Speech
- **pygame**: Audio playback

---

For any questions or issues, feel free to contact or submit an issue on GitHub.