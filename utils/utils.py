import numpy as np
from pedalboard.io import AudioFile
from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, Gain
import noisereduce as nr
from moviepy import AudioFileClip, VideoClip
from PIL import Image, ImageDraw
import matplotlib.colors as mcolors
import re


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
    output_gain: float = 10,
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
            prop_decrease=noise_prop_decrease,
        )

    board = Pedalboard(
        [
            NoiseGate(
                threshold_db=gate_threshold, ratio=gate_ratio, release_ms=gate_release
            ),
            Compressor(threshold_db=comp_threshold, ratio=comp_ratio),
            LowShelfFilter(
                cutoff_frequency_hz=low_shelf_cutoff, gain_db=low_shelf_gain
            ),
            Gain(gain_db=output_gain),
        ]
    )

    effected = board(audio, sample_rate)

    with AudioFile(output_path, "w", sample_rate, effected.shape[0]) as f:
        f.write(effected)


def validate_color(rgba):
    """
    Convert an RGBA color (with values in 0-255 range) to HEX format.
    If the input is already a valid hex color, return it directly.

    Args:
        rgba (tuple or str): A tuple (R, G, B, A) or a string like 'rgba(R, G, B, A)' or a hex color string.

    Returns:
        str: Hex color string.
    """

    # Check if the input is already a valid hex color
    if isinstance(rgba, str) and re.match(r"^#(?:[0-9a-fA-F]{3}){1,2}$", rgba):
        return rgba

    if isinstance(rgba, str):
        # Extract numbers from 'rgba(r, g, b, a)' string format
        rgba = tuple(map(float, rgba.strip("rgba()").split(",")))

    if not isinstance(rgba, (tuple, list)) or len(rgba) != 4:
        raise ValueError("Invalid RGBA format. Expected a tuple (R, G, B, A)")

    rgb = tuple(int(round(c)) for c in rgba[:3])  # Convert R, G, B to integers (0-255)
    hex_color = mcolors.to_hex([c / 255 for c in rgb])  # Normalize to 0-1 range

    return hex_color


def generate_waveform_video(
    input_audio_path: str,
    output_video_path: str,
    sample_rate: int = 44100,
    fps: int = 30,
    resolution: tuple = (640, 480),
    window_duration: float = 0.1,
    line_color: tuple = (255, 255, 255),  # White waveform
    bg_color: tuple = (0, 0, 0),  # Black background
):
    """
    Generate a waveform video visualization for the given audio file.

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
    # Load the audio file
    audio_clip = AudioFileClip(input_audio_path)
    audio = audio_clip.to_soundarray(fps=sample_rate)

    # Convert to mono if audio has more than one channel
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Normalize audio to [-1, 1] range
    audio = audio / np.max(np.abs(audio))

    total_duration = audio_clip.duration
    width, height = resolution

    def make_frame(t: float):
        """
        Generate a single video frame at time t.
        """
        # Convert time into the corresponding sample index
        center_sample = int(t * sample_rate)
        half_window = int((window_duration * sample_rate) / 2)
        start = max(center_sample - half_window, 0)
        end = min(center_sample + half_window, len(audio))

        # Extract the segment and pad if necessary
        segment = audio[start:end]
        if len(segment) < (2 * half_window):
            segment = np.pad(segment, (0, (2 * half_window) - len(segment)), "constant")

        # Resample the segment to match the horizontal resolution of the video frame
        indices = np.linspace(0, len(segment) - 1, num=width).astype(int)
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

        return np.array(img)

    # Create the video clip using the frame maker function
    video_clip = VideoClip(make_frame, duration=total_duration).with_fps(fps)

    # Set the audio track so that the video syncs with the original audio
    video_clip = video_clip.with_audio(audio_clip)

    # Write the video out to file with the specified frames per second
    video_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
