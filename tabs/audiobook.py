import gradio as gr
import numpy as np

def text_to_audiobook(file, voice, language, speed, kokoro):
    try:
        if not file:
            raise ValueError("Please upload a text file")
            
        with open(file.name, 'r') as f:
            text = f.read()
            
        if not text.strip():
            raise ValueError("The uploaded file is empty")
            
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        audio_chunks = []
        sample_rate = None
        
        for paragraph in paragraphs:
            # Process in chunks to avoid memory issues
            samples, sr = kokoro.create(
                text=paragraph,
                voice=voice,
                speed=speed,
                lang=language
            )
            sample_rate = sr
            audio_chunks.append(samples)
            
            # Add short pause between paragraphs
            silence = np.zeros(int(0.5 * sr))  # 0.5 second pause
            audio_chunks.append(silence)
            
        final_audio = np.concatenate(audio_chunks)
        return (sample_rate, final_audio)
        
    except Exception as e:
        print(f"Audiobook generation error: {e}")
        raise gr.Error(f"Failed to generate audiobook: {str(e)}")

def create_audiobook_tab(kokoro, available_voices, available_languages):
    with gr.Tab("Audiobook"):
        gr.Markdown("""
        ### Generate Audiobook from Text File
        Upload a .txt file and convert its contents to audiobook format.
        """)
        
        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="Text File",
                    type="filepath",
                    file_types=[".txt"]
                )
                
                voice_input = gr.Dropdown(
                    choices=available_voices,
                    value=available_voices[0],
                    label="Narrator Voice"
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
                    label="Reading Speed"
                )
                
                generate_btn = gr.Button("Generate Audiobook", variant="primary")
            
            with gr.Column():
                audio_output = gr.Audio(
                    label="Generated Audiobook",
                    type="numpy"
                )
        
        generate_btn.click(
            fn=lambda *args: text_to_audiobook(*args, kokoro),
            inputs=[file_input, voice_input, language_input, speed_input],
            outputs=audio_output
        ) 