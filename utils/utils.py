import numpy as np
from pedalboard.io import AudioFile
from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, Gain
import noisereduce as nr

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
