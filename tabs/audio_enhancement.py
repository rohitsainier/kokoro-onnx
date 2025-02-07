import os
import gradio as gr
import tempfile
from utils.utils import process_audio_enhancement

# Imports for generating spectrograms
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

def preview_input_audio(input_file):
    # A simple callback function to enable previewing the uploaded audio file.
    # It returns the file path that the gr.Audio component can use to play the audio.
    return input_file

def generate_spectrogram(audio_file_path):
    """
    Generate a spectrogram image from an audio file.
    Returns the file path to the spectrogram image.
    """
    try:
        # Load audio with original sampling rate
        y, sr = librosa.load(audio_file_path, sr=None)
    except Exception as e:
        raise gr.Error(f"Error loading audio file for spectrogram: {str(e)}")
    
    # Generate mel spectrogram and convert to decibels.
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Create the plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.title('Mel-frequency spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    
    # Save the spectrogram image to a temporary file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        output_img = tmpfile.name
    plt.savefig(output_img)
    plt.close()
    
    return output_img

def enhance_audio(
    input_file,
    sample_rate,
    noise_reduction,
    noise_stationary,
    noise_prop_decrease,
    gate_threshold,
    gate_ratio,
    gate_release,
    comp_threshold,
    comp_ratio,
    low_shelf_cutoff,
    low_shelf_gain,
    output_gain
):
    if input_file is None:
        raise gr.Error("Please upload an audio file.")
    
    # Create a temporary file for output
    tmp_dir = tempfile.gettempdir()
    output_file = os.path.join(tmp_dir, "enhanced_audio.wav")
    
    try:
        process_audio_enhancement(
            input_path=input_file,
            output_path=output_file,
            sample_rate=sample_rate,
            noise_reduction=noise_reduction,
            noise_stationary=noise_stationary,
            noise_prop_decrease=noise_prop_decrease,
            gate_threshold=gate_threshold,
            gate_ratio=gate_ratio,
            gate_release=gate_release,
            comp_threshold=comp_threshold,
            comp_ratio=comp_ratio,
            low_shelf_cutoff=low_shelf_cutoff,
            low_shelf_gain=low_shelf_gain,
            output_gain=output_gain
        )
    except Exception as e:
        raise gr.Error(f"Error processing audio: {str(e)}")
    
    # Generate spectrograms for both input and enhanced audio files.
    input_spectrogram = generate_spectrogram(input_file)
    output_spectrogram = generate_spectrogram(output_file)
    
    # Return tuple: enhanced audio file, input spectrogram image, output spectrogram image.
    return output_file, input_spectrogram, output_spectrogram

def create_audio_enhancement_tab():
    with gr.Tab("Audio Enhancement"):
        gr.Markdown(
            """
            ### Enhance Your Audio File
            
            Upload an audio file and adjust the parameters to reduce noise, compress, and apply shelf filtering.
            You can also compare the spectrogram of your input audio with that of the enhanced output.
            """
        )
        with gr.Row():
            # Left column: input controls
            with gr.Column():
                file_input = gr.File(label="Upload Audio File", type="filepath", file_types=[".wav", ".mp3", ".aiff"])
                audio_input_preview = gr.Audio(label="Input Audio Preview", type="filepath")
                file_input.change(fn=preview_input_audio, inputs=file_input, outputs=audio_input_preview)

                sample_rate_input = gr.Slider(
                    minimum=8000, maximum=48000, step=1000, value=44100, label="Sample Rate"
                )
                noise_reduction_input = gr.Checkbox(value=True, label="Apply Noise Reduction")
                noise_stationary_input = gr.Checkbox(value=True, label="Stationary Noise Reduction")
                noise_prop_input = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.05, value=0.75, label="Noise Prop Decrease"
                )
                gate_threshold_input = gr.Slider(
                    minimum=-60, maximum=0, step=1, value=-30, label="Gate Threshold (dB)"
                )
                gate_ratio_input = gr.Slider(
                    minimum=1.0, maximum=3.0, step=0.1, value=1.5, label="Gate Ratio"
                )
                gate_release_input = gr.Slider(
                    minimum=50, maximum=500, step=10, value=250, label="Gate Release (ms)"
                )
                comp_threshold_input = gr.Slider(
                    minimum=-40, maximum=0, step=1, value=-16, label="Compressor Threshold (dB)"
                )
                comp_ratio_input = gr.Slider(
                    minimum=1.0, maximum=4.0, step=0.1, value=2.5, label="Compressor Ratio"
                )
                low_shelf_cutoff_input = gr.Slider(
                    minimum=100, maximum=1000, step=10, value=400, label="Low Shelf Cutoff (Hz)"
                )
                low_shelf_gain_input = gr.Slider(
                    minimum=0, maximum=20, step=1, value=10, label="Low Shelf Gain (dB)"
                )
                output_gain_input = gr.Slider(
                    minimum=0, maximum=20, step=1, value=10, label="Output Gain (dB)"
                )
                enhance_btn = gr.Button("Enhance Audio", variant="primary")
            
            # Right column: output components (enhanced audio at top, spectrograms below)
            with gr.Column():
                audio_output = gr.Audio(label="Enhanced Audio", type="filepath")
                with gr.Row():
                    with gr.Column():
                        spectrogram_input_image = gr.Image(label="Input Audio Spectrogram", type="filepath")
                        spectrogram_output_image = gr.Image(label="Enhanced Audio Spectrogram", type="filepath")
        
        # Connect all inputs to the enhancement function
        enhance_btn.click(
            fn=enhance_audio,
            inputs=[
                file_input,
                sample_rate_input,
                noise_reduction_input,
                noise_stationary_input,
                noise_prop_input,
                gate_threshold_input,
                gate_ratio_input,
                gate_release_input,
                comp_threshold_input,
                comp_ratio_input,
                low_shelf_cutoff_input,
                low_shelf_gain_input,
                output_gain_input,
            ],
            outputs=[audio_output, spectrogram_input_image, spectrogram_output_image],
        ) 