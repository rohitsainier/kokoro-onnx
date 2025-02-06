import gradio as gr
import requests
import ollama
import tempfile
import soundfile as sf
import whisper

def get_ollama_response(prompt: str, model: str) -> str:
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response['response']
    except Exception as e:
        print(f"Error getting Ollama response: {e}")
        return f"Error: {str(e)}"

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

def playground_pipeline(prompt: str, model: str, voice: str, language: str, speed: float, kokoro):
    try:
        # Check for empty prompt
        if not prompt or prompt.strip() == "":
            return "Please provide some text input", None
            
        if model.startswith("Error") or model.startswith("No models"):
            return "Please install and select a valid model first", None
            
        ai_response = get_ollama_response(prompt, model)
        
        # Check for empty AI response
        if not ai_response or ai_response.strip() == "":
            return "Received empty response from AI model", None
        
        audio_array, sample_rate = kokoro.create(
            text=ai_response,
            voice=voice,
            speed=speed,
            lang=language
        )
        
        return ai_response, (sample_rate, audio_array)
    except Exception as e:
        print(f"Error in playground pipeline: {e}")
        return str(e), None

def speech_to_text(audio):
    """
    Convert speech to text using Whisper
    Args:
        audio: Audio file or array from microphone
    Returns:
        str: Transcribed text
    """
    try:
        if audio is None:
            return ""
            
        # Save the audio array to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
            sf.write(temp_audio.name, audio[1], audio[0], format='WAV')
            
            # Load Whisper model and transcribe
            model = whisper.load_model("base")
            result = model.transcribe(temp_audio.name)
            
            return result["text"].strip()
    except Exception as e:
        print(f"Error in speech to text: {e}")
        return f"Error: {str(e)}"

def create_playground_tab(kokoro, available_voices, available_languages):
    with gr.Tab("Playground"):
        gr.Markdown("""
        ### AI Text-to-Speech Playground
        Speak into the microphone and the system will automatically transcribe, generate a response, and play it back.
        Make sure you have Ollama running locally with at least one model installed.
        
        To install a model: `ollama pull <model_name>`
        Example: `ollama pull mistral`
        """)
        
        with gr.Row():
            with gr.Column():
                model_input = gr.Dropdown(
                    choices=get_available_models(),
                    label="AI Model",
                    info="Select an available model from your local Ollama instance",
                    allow_custom_value=False,
                    interactive=True
                )
                
                refresh_models = gr.Button("🔄 Refresh Models", size="sm")
                
                # Modified microphone input with auto_submit=True
                audio_input = gr.Audio(
                    label="Input Audio",
                    type="numpy",
                    interactive=True,
                    sources=["microphone"],
                    streaming=False,
                    autoplay=False
                )
                
                prompt_input = gr.Textbox(
                    label="Your Prompt",
                    placeholder="Enter your prompt here or use the microphone above...",
                    lines=5
                )
                
                voice_input = gr.Dropdown(
                    choices=available_voices,
                    value=available_voices[0],
                    label="Voice"
                )
                
                language_input = gr.Dropdown(
                    choices=available_languages,
                    value="en-us",
                    label="Language"
                )
                
                speed_input = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Speech Speed"
                )
            
            with gr.Column():
                response_output = gr.Textbox(
                    label="AI Response",
                    lines=10,
                    interactive=False
                )
                
                audio_output = gr.Audio(
                    label="Generated Speech",
                    type="numpy",
                    interactive=False,
                    autoplay=True  # Enable autoplay for output audio
                )
        
        def refresh_model_list():
            return gr.Dropdown(choices=get_available_models())
        
        refresh_models.click(
            fn=refresh_model_list,
            outputs=model_input
        )
        
        # Create an automatic pipeline function
        def auto_pipeline(audio, model, voice, language, speed):
            # First transcribe
            transcribed_text = speech_to_text(audio)
            
            # Then generate response and audio
            response, audio_out = playground_pipeline(transcribed_text, model, voice, language, speed, kokoro)
            
            return transcribed_text, response, audio_out
        
        # Remove the separate transcribe button and use audio_input's change event
        audio_input.change(
            fn=auto_pipeline,
            inputs=[
                audio_input,
                model_input,
                voice_input,
                language_input,
                speed_input
            ],
            outputs=[
                prompt_input,
                response_output,
                audio_output
            ]
        )
        
        # Keep the generate button for text input cases
        generate_btn = gr.Button("Generate Response & Audio", variant="primary")
        generate_btn.click(
            fn=lambda *args: playground_pipeline(*args, kokoro),
            inputs=[prompt_input, model_input, voice_input, language_input, speed_input],
            outputs=[response_output, audio_output]
        ) 

