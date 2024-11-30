import os
import pygame
import streamlit as st
from gtts import gTTS
from pathlib import Path
import glob
import fitz
import atexit

def ensure_audio_file_exists(AUDIO_DIR, filename):
    """Create an empty audio file if it doesn't exist"""
    filepath = os.path.join(AUDIO_DIR, filename)
    Path(filepath).touch(exist_ok=True)
    return filepath

def cleanup_session_files(AUDIO_DIR):
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

def text_to_audio(text, AUDIO_DIR, filename):
    filepath = os.path.join(AUDIO_DIR, filename)
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filepath)
        return filepath
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None
    
def cleanup_temp_files(AUDIO_DIR, pattern="page_*.mp3"):
    """Clean up temporary audio files"""
    for file in glob.glob(os.path.join(AUDIO_DIR, pattern)):
        os.remove(file)

def voice_read_pdf(pages, AUDIO_DIR, start_page=0):
    """Voice read PDF pages with play/pause/resume functionality"""
    if not pygame.mixer.get_init():
        pygame.mixer.init()

    if 'reading_state' not in st.session_state:
        st.session_state.reading_state = {
            'current_page': start_page,
            'is_playing': False,
            'audio_files': []
        }

    if not st.session_state.reading_state['audio_files']:
        cleanup_temp_files(AUDIO_DIR)
        for idx, page in enumerate(pages[start_page:], start=start_page):
            audio_file = text_to_audio(page, AUDIO_DIR, f"page_{idx}.mp3")
            st.session_state.reading_state['audio_files'].append(audio_file)

    current_page = st.session_state.reading_state['current_page']
    
    st.text_area("Current Page Content", pages[current_page], height=300)

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
            if block['type'] == 0:
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
