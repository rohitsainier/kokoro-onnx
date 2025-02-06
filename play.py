import requests
import ollama
import soundfile as sf
import whisper
import numpy as np
import pyaudio
import os
import wave
from app import init_kokoro  # Import the kokoro initialization method from app.py
import threading

# Initialize Kokoro using app.py
kokoro = init_kokoro()

def get_available_models() -> list[str]:
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = [model['name'] for model in response.json()['models']]
            if not models:
                return ["No models found - please pull a model first using 'ollama pull <model_name>'"]
            return sorted(models)
        else:
            return ["Error connecting to Ollama"]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return ["Error: Is Ollama server running? Start with 'ollama serve'"]

def get_ollama_response(prompt: str, model: str) -> str:
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response['response']
    except Exception as e:
        print(f"Error getting Ollama response: {e}")
        return f"Error: {str(e)}"

def record_audio(file_path):
    """
    Record audio from the microphone and save to a file.
    Press Enter to stop recording.
    """
    # Event to notify when the user presses Enter.
    stop_recording_event = threading.Event()

    # This function will block waiting for an Enter key, then signal to stop.
    def wait_for_enter():
        input("\n")
        stop_recording_event.set()

    # Start the input-waiting thread.
    input_thread = threading.Thread(target=wait_for_enter)
    input_thread.start()

    # Setup PyAudio for recording.
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                    input=True, frames_per_buffer=1024)
    frames = []

    print("Recording...")

    # Read audio data in chunks until the stop event is signaled.
    while not stop_recording_event.is_set():
        data = stream.read(1024)
        frames.append(data)

    print("Recording stopped.")

    stream.stop_stream()
    stream.close()
    p.terminate()
    input_thread.join()

    # Save the recorded audio to a file.
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def play_audio(file_path):
    # Open the audio file
    wf = wave.open(file_path, 'rb')
    
    # Create a PyAudio instance
    p = pyaudio.PyAudio()
    
    # Open a stream to play audio
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    
    # Read and play audio data
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    
    # Stop and close the stream and PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()

def transcribe_with_whisper(audio_file_path):
    # Load the model
    model = whisper.load_model("base.en")
    result = model.transcribe(audio_file_path)
    return result["text"]

def generate_audio(text: str, voice: str, speed: float, language: str, output_file: str = "generated_audio.wav") -> str:
    try:
        # Use the kokoro instance imported from app.py to generate audio from text
        audio_array, sample_rate = kokoro.create(
            text=text,
            voice=voice,
            speed=speed,
            lang=language
        )
        
        # Convert the audio array from float32 to int16 if needed.
        if audio_array is not None and audio_array.dtype == np.float32:
            audio_array = (audio_array * 32767).astype(np.int16)
        
        # Write the audio array to a file using soundfile.
        sf.write(output_file, audio_array, sample_rate)
        
        return output_file
    except Exception as e:
        print(f"Error generating audio file: {e}")
        raise e

def terminal_chatbot_conversation():
    """
    Audio-based chatbot conversation:
    - Records user spoken input and transcribes it into text.
    - Sends the text to the Ollama model for a response.
    - Generates audio from the assistant's text response and plays it.
    - To stop speaking during the recording, press CTRL+C.
    - Say "exit" during your speech to terminate the conversation.
    """
    conversation_history = []
    
    # Retrieve the first available model.
    models = get_available_models()
    if models and not models[0].startswith("Error"):
        try:
            model = models[1]
        except IndexError:
            model = models[0]
        print(f"Using model: {model}")
    else:
        print("No valid model available - please pull a model and ensure the Ollama server is running.")
        return
    
    print("Starting audio conversation. Say 'exit' to quit the conversation.\n")
    
    while True:
        input_audio_file = "user_input.wav"
        
        # Record user audio from the microphone.
        try:
            record_audio(input_audio_file)
        except Exception as e:
            print(f"Error recording audio: {e}")
            continue
        
        # Transcribe the recorded audio using Whisper.
        try:
            user_input = transcribe_with_whisper(input_audio_file).strip()
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            continue
        
        # Print user transcription in green.
        print("\033[1;32mYou: " + user_input + "\033[0m")
        
        if user_input.lower() == "exit":
            print("Exiting conversation.")
            break
        
        conversation_history.append(f"You: {user_input}")
        
        # Get assistant response from the Ollama model.
        assistant_response = get_ollama_response(user_input, model)
        conversation_history.append(f"Assistant: {assistant_response}")
        
       # Display the assistant text response in blue.
        print("\033[1;34mSky:\033[0m " + "\033[1;34m" + assistant_response + "\033[0m")
        
        # Generate audio for the assistant's text response and play it.
        try:
            assistant_audio_file = "assistant_response.wav"
            generate_audio(assistant_response, voice="af_sky", speed=1.0, language="en-us", output_file=assistant_audio_file)
            play_audio(assistant_audio_file)
        except Exception as e:
            print(f"Error generating or playing assistant audio: {e}")
        
        # Limit conversation history to the last 20 messages.
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

if __name__ == "__main__":
    terminal_chatbot_conversation()