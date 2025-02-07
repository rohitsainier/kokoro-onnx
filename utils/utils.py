import numpy as np
from pedalboard.io import AudioFile
from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, Gain
import noisereduce as nr
from PIL import Image, ImageDraw
from moviepy import *

def process_audio_enhancement(
    input_path: str,
    output_path: str,
    sample_rate: int = 44100,
    noise_reduction: bool = True,
    noise_stationary: bool = True,
    noise_prop_decrease: float = 0.75,
    gate_threshold: float = -30,
    gate_ratio: float = 1.5,
    gate_release: float = 250,
    comp_threshold: float = -16,
    comp_ratio: float = 2.5,
    low_shelf_cutoff: float = 400,
    low_shelf_gain: float = 10,
    output_gain: float = 10
):
    """
    Process audio file with noise reduction and effects chain.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save processed audio file
        sample_rate: Target sample rate (default: 44100)
        Other parameters control individual processing steps
    """
    with AudioFile(input_path).resampled_to(sample_rate) as f:
        audio = f.read(f.frames)

    if noise_reduction:
        audio = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=noise_stationary,
            prop_decrease=noise_prop_decrease
        )

    board = Pedalboard([
        NoiseGate(threshold_db=gate_threshold, ratio=gate_ratio, release_ms=gate_release),
        Compressor(threshold_db=comp_threshold, ratio=comp_ratio),
        LowShelfFilter(cutoff_frequency_hz=low_shelf_cutoff, gain_db=low_shelf_gain),
        Gain(gain_db=output_gain)
    ])

    effected = board(audio, sample_rate)

    with AudioFile(output_path, 'w', sample_rate, effected.shape[0]) as f:
        f.write(effected)


def generate_waveform_video(
    input_audio_path: str,
    output_video_path: str,
    sample_rate: int = 44100,
    fps: int = 30,
    resolution: tuple = (640, 480),
    window_duration: float = 0.1,
    line_color: tuple = (255, 255, 255),  # white waveform
    bg_color: tuple = (0, 0, 0)             # black background
):
    """
    Generate a waveform video visualization for the given audio file.
    
    The video displays a moving waveform visualization of the audio; each frame shows a segment
    of the audio, so that if the audio is a sine wave the visualized waveform will be a sine wave
    with matching frequency. The generated video is synchronized with the original audio track.
    
    Args:
        input_audio_path (str): Path to the input audio file.
        output_video_path (str): Path where the generated video will be saved.
        sample_rate (int): Sample rate for reading the audio (default: 44100).
        fps (int): Frames per second for the video (default: 30).
        resolution (tuple): (width, height) in pixels for the video resolution (default: (640, 480)).
        window_duration (float): Duration (in seconds) of the audio segment to visualize in each frame.
        line_color (tuple): RGB tuple for the color of the waveform line (default: white).
        bg_color (tuple): RGB tuple for the background color of the video (default: black).
    """
    # Import additional packages required for video generation and drawing.
    

    # Read the entire audio file at the given sample rate
    with AudioFile(input_audio_path).resampled_to(sample_rate) as f:
        audio = f.read(f.frames)

    # Convert to mono if audio has more than one channel
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Normalize audio to [-1, 1] range if not already
    audio = audio / np.max(np.abs(audio))

    total_duration = len(audio) / sample_rate
    width, height = resolution

    def make_frame(t: float):
        """
        Generate a single video frame at time t.
        """
        # Convert time into the corresponding sample index
        center_sample = int(t * sample_rate)
        half_window = int((window_duration * sample_rate) / 2)
        start = center_sample - half_window
        end = center_sample + half_window

        # If the window extends beyond the available samples, pad with zeros
        if start < 0:
            segment = np.pad(audio[0:end], (abs(start), 0), mode='constant')
        elif end > len(audio):
            segment = np.pad(audio[start:len(audio)], (0, end - len(audio)), mode='constant')
        else:
            segment = audio[start:end]

        # Resample the segment to match the horizontal resolution of the video frame
        segment_length = len(segment)
        indices = np.linspace(0, segment_length - 1, num=width).astype(int)
        waveform = segment[indices]

        # Scale the waveform: center vertically with amplitude scaled to half the height
        middle = height // 2
        amplitude = height // 2 * 0.95  # Add 5% padding
        ys = middle - (waveform * amplitude).astype(int)

        # Create a blank image
        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        # Draw the waveform by connecting points along the horizontal axis
        points = [(x, int(ys[x])) for x in range(width)]
        draw.line(points, fill=line_color, width=2)

        # Optional: Draw a vertical red line in the center as a time indicator
        draw.line([(width // 2, 0), (width // 2, height)], fill=(255, 0, 0), width=1)

        return np.array(img)

    # Create the video clip using the frame maker function
    video_clip = VideoClip(make_frame, duration=total_duration)
    # Set the audio track so that the video syncs with the original audio
    video_clip = video_clip.with_audio(AudioFileClip(input_audio_path))
    
    # Write the video out to file with the specified frames per second
    video_clip.write_videofile(output_video_path, fps=fps)
