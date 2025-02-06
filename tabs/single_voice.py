import gradio as gr

def text_to_speech(text, voice, language, speed, kokoro):
    try:
        audio_array, sample_rate = kokoro.create(
            text=text,
            voice=voice,
            speed=speed,
            lang=language
        )
        return (sample_rate, audio_array)
    except Exception as e:
        return None

def create_single_voice_tab(kokoro, available_voices, available_languages):
    with gr.Tab("Single Voice"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text to speak",
                    placeholder="Enter the text you want to convert to speech...",
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
                    label="Speed"
                )
                
                submit_btn = gr.Button("Generate Speech", variant="primary")
            
            with gr.Column():
                audio_output = gr.Audio(
                    label="Generated Speech",
                    type="numpy",
                    interactive=False
                )
        
        submit_btn.click(
            fn=lambda *args: text_to_speech(*args, kokoro),
            inputs=[text_input, voice_input, language_input, speed_input],
            outputs=audio_output
        ) 