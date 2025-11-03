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

def load_tracklist(file_path):
    tracklist = []
    with open(file_path, "r") as f:
        for line in f:
            if "\t" not in line:
                continue
            title, time_str = line.strip().split("\t")
            h, m, s = (["0"] * (3 - len(time_str.split(":"))) + time_str.split(":"))
            total_seconds = int(h) * 3600 + int(m) * 60 + int(s)
            tracklist.append((title, total_seconds))
    return tracklist

def animated_position(t, t_in, t_out, duration=1.0, x_start=-600, x_end=200):
    """
    Returns current x offset for a fly-in/fly-out animation.
    t = current time (s)
    t_in = animation start time (s)
    t_out = animation exit start (s)
    duration = duration of fly-in and fly-out (s)
    """
    if t < t_in:
        return None  # not started
    elif t_in <= t < t_in + duration:
        # fly in
        progress = (t - t_in) / duration
        return int(x_start + (x_end - x_start) * (0.5 - 0.5 * np.cos(np.pi * progress)))
    elif t_in + duration <= t < t_out:
        # hold position
        return int(x_end)
    elif t_out <= t < t_out + duration:
        # fly out
        progress = (t - t_out) / duration
        return int(x_end + (1920 - x_end) * (0.5 - 0.5 * np.cos(np.pi * progress)))
    else:
        return None
