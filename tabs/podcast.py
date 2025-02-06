import gradio as gr
import pandas as pd
import numpy as np
import scipy.signal as signal
from kokoro_onnx import Kokoro

def create_podcast(kokoro, available_voices, sentences):
    try:
        audio = []
        sample_rate = None
        
        # Convert dataframe rows to list of dictionaries
        for i in range(len(sentences)):
            # Get row as dictionary
            row = sentences.iloc[i]
            voice = row['voice']
            text = row['text']
            
            # Handle potential NaN values from empty rows
            if pd.isna(voice) or pd.isna(text):
                print(f"Skipping row {i+1} due to empty voice or text")
                continue  # Skip empty rows
            
            # Validate input
            if not voice or voice not in available_voices:
                raise ValueError(f"Invalid voice '{voice}' at row {i+1}")
            if not text.strip():
                raise ValueError(f"Empty text at row {i+1}")
            
            # Generate audio for each sentence
            samples, sr = kokoro.create(
                text=text,
                voice=voice,
                speed=1.0,
                lang="en-us"
            )
            sample_rate = sr
            
            # Convert mono to stereo by duplicating channel
            stereo_samples = np.column_stack((samples, samples))
            audio.append(stereo_samples)
            
            # Add random pause between sentences (also in stereo)
            silence_duration = np.random.uniform(0.5, 1.0)
            silence = np.zeros((int(silence_duration * sr), 2))
            audio.append(silence)
        
        # Concatenate all audio parts
        final_audio = np.concatenate(audio)
        # Apply default Audacity EQ to enhance bass and treble
        #final_audio = apply_audacity_eq(final_audio, sample_rate)
        return (sample_rate, final_audio)
    except Exception as e:
        print(f"Error creating podcast: {e}")
        return None

def create_podcast_tab(kokoro, available_voices, EXAMPLE_SENTENCES):
    with gr.Tab("Podcast"):
        gr.Markdown("""
        ### Create a Multi-Voice Podcast
        Add speakers and their lines to create a natural-sounding conversation.
        Each row represents one segment of speech.
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Template for new entries
                with gr.Row():
                    new_voice = gr.Dropdown(
                        choices=available_voices,
                        value=available_voices[0],
                        label="Voice",
                        scale=1,
                        interactive=True,
                        allow_custom_value=False
                    )
                    new_text = gr.Textbox(
                        label="Text",
                        placeholder="Enter what this voice should say...",
                        scale=2
                    )
                    add_btn = gr.Button("Add Line", size="sm")
                
                # Add file upload component
                with gr.Row():
                    file_upload = gr.UploadButton(
                        "Upload TXT Script",
                        file_types=[".txt"],
                        file_count="single",
                        variant="secondary",
                        size="sm"
                    )
                
                podcast_input = gr.Dataframe(
                    headers=["voice", "text"],
                    datatype=["str", "str"],
                    label="Podcast Script",
                    value=pd.DataFrame(columns=["voice", "text"]),
                    interactive=True,
                    wrap=True
                )
                
                with gr.Row():
                    clear_btn = gr.Button("Clear All", variant="secondary", size="sm")
                    example_btn = gr.Button("Load Example", variant="secondary", size="sm")
                    podcast_btn = gr.Button("Generate Podcast", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                podcast_output = gr.Audio(
                    label="Generated Podcast",
                    type="numpy"
                )
                gr.Markdown("""
                ### Tips:
                - Use different voices to create engaging conversations
                - Short, natural sentences work best
                - The system automatically adds small pauses between lines
                - You can edit or delete rows directly in the table
                """)
        
        # Button handlers
        def add_line(voice, text, existing_data):
            if not voice or voice not in available_voices:
                raise gr.Error("Please select a valid voice from the dropdown")
            if not text.strip():
                raise gr.Error("Please enter some text for the speech line")
            
            new_row = pd.DataFrame([[voice, text]], columns=existing_data.columns)
            return pd.concat([existing_data, new_row], ignore_index=True)
        
        def clear_table():
            return []
        
        def load_example():
            example_data = [[s["voice"], s["text"]] for s in EXAMPLE_SENTENCES]
            return pd.DataFrame(example_data, columns=["voice", "text"])
        
        add_btn.click(
            fn=add_line,
            inputs=[new_voice, new_text, podcast_input],
            outputs=[podcast_input]
        )
        
        clear_btn.click(
            fn=clear_table,
            inputs=None,
            outputs=[podcast_input]
        )
        
        example_btn.click(
            fn=load_example,
            inputs=None,
            outputs=[podcast_input]
        )
        
        podcast_btn.click(
            fn=lambda df: create_podcast(kokoro, available_voices, df),
            inputs=[podcast_input],
            outputs=[podcast_output]
        )

        # Add file upload handler
        def handle_file_upload(file):
            try:
                with open(file.name, "r", encoding="utf-8") as f:
                    content = f.read()
                
                data = []
                for line_num, line in enumerate(content.splitlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    if ":" not in line:
                        raise gr.Error(f"Line {line_num}: Missing colon separator")
                    voice_part, _, text_part = line.partition(":")
                    voice = voice_part.strip().lower()
                    text = text_part.strip()
                    
                    if voice not in available_voices:
                        raise gr.Error(f"Line {line_num}: Voice '{voice}' not in available voices")
                    if not text:
                        raise gr.Error(f"Line {line_num}: Empty text after colon")
                    
                    data.append([voice, text])
                
                return pd.DataFrame(data, columns=["voice", "text"])
            except Exception as e:
                raise gr.Error(f"File processing error: {str(e)}")
        
        file_upload.upload(
            fn=handle_file_upload,
            inputs=[file_upload],
            outputs=[podcast_input],
        )