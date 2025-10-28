import numpy as np
from moviepy import ImageSequenceClip, AudioFileClip
from scipy.interpolate import interp1d



def create_video(music_file, frames_folder, output_file, fps=30):
    # Create video from frames and add audio
    audio = AudioFileClip(music_file)
    clip = ImageSequenceClip(frames_folder, fps=fps)
    clip = clip.with_audio(audio)  # Add audio to the clip after creation
    clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

    print(f"Video file {output_file} created successfully.")


def smooth(values, window=5):
    return np.convolve(values, np.ones(window)/window, mode='same')


def normalize_and_smooth(values):
    """Smooth, interpolate, normalize, and floor the amplitude values."""
    x = np.arange(len(values))

    # Protect against -inf/NaN and require some minimum value to treat as valid
    values = np.nan_to_num(values, neginf=-120.0)
    mask = np.isfinite(values) & (values > -70)  # ignore very silent bands

    # Handle cases where interp1d would get empty or single-point input
    if mask.sum() == 0:
        # no valid points -> use a quiet constant
        values = np.full_like(values, -70.0)
    elif mask.sum() == 1:
        # only one point -> fill with that single value (no interpolation possible)
        single_val = values[mask][0]
        values = np.full_like(values, single_val)
    else:
        # Interpolate missing bins for smooth continuity
        interp = interp1d(x[mask], values[mask], kind='linear', fill_value="extrapolate")
        values = interp(x)

    # Smooth
    values = smooth(values, window=7)

    # Normalize per frame
    values -= values.min()
    if values.max() > 0:
        values /= values.max()

    # Scale to pixel size and floor minimum bar length
    values = values * 200
    values = np.maximum(values, 20)

    return values
