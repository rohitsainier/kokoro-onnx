import gradio as gr
from kokoro_onnx import Kokoro
import os
import wget
import numpy as np
import pandas as pd
import onnxruntime
from onnxruntime import InferenceSession
import ollama
import requests
import whisper
import tempfile
import soundfile as sf

from tabs import (
    create_single_voice_tab,
    create_podcast_tab,
    create_audiobook_tab,
    create_playground_tab
)

# Constants
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
MODEL_PATH = "kokoro-v1.0"
VOICES_PATH = "voices-v1.0"
EXAMPLE_SENTENCES = [
    {"voice": "af_sarah", "text": "Hello and welcome to the podcast! We've got some exciting things lined up today."},
    {"voice": "am_michael", "text": "It's going to be an exciting episode. Stick with us!"},
    {"voice": "af_sarah", "text": "But first, we've got a special guest with us. Please welcome Nicole!"},
    {"voice": "af_sarah", "text": "Now, we've been told Nicole has a very unique way of speaking today... a bit of a mysterious vibe, if you will."},
    {"voice": "af_nicole", "text": "Hey there... I'm so excited to be a guest today... But I thought I'd keep it quiet... for now..."},
    {"voice": "am_michael", "text": "Well, it certainly adds some intrigue! Let's dive in and see what that's all about."},
    {"voice": "af_sarah", "text": "Today, we're covering something that's close to our hearts"},
    {"voice": "am_michael", "text": "It's going to be a good one!"}
]

def download_models():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        wget.download(MODEL_URL, MODEL_PATH)
    
    if not os.path.exists(VOICES_PATH):
        print("Downloading voices...")
        wget.download(VOICES_URL, VOICES_PATH)

def init_kokoro():
    download_models()
    session = InferenceSession(
        MODEL_PATH,
        providers=['CPUExecutionProvider']
    )
    return Kokoro.from_session(session, VOICES_PATH)

def create_ui():
    # Initialize Kokoro
    kokoro = init_kokoro()
    
    # Get available voices and languages
    available_voices = kokoro.get_voices()
    available_languages = kokoro.get_languages()
    
    # Create the Gradio interface
    with gr.Blocks(title="Kokoro TTS") as interface:
        gr.Markdown("# Kokoro Text-to-Speech")
        
        with gr.Tabs():
            create_single_voice_tab(kokoro, available_voices, available_languages)
            create_podcast_tab(kokoro, available_voices, EXAMPLE_SENTENCES)
            create_audiobook_tab(kokoro, available_voices, available_languages)
            create_playground_tab(kokoro, available_voices, available_languages)
        
        gr.Markdown("""
        ### Usage Instructions:
        1. Select a tab based on your needs:
           - Single Voice: Convert simple text to speech
           - Podcast: Create multi-speaker conversations
           - Audiobook: Convert longer texts with paragraph support
           - Playground: Experiment with various voice settings
        
        2. Enter your text and adjust the settings
        3. Click the generate button to create speech
        4. Use the audio player to listen to the result
        
        Note: Processing time may vary based on text length and system capabilities.
        """)
    
    return interface

if __name__ == "__main__":
    interface = create_ui()
    interface.launch(share=False)