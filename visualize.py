import librosa
import numpy as np
from moviepy import ImageSequenceClip, AudioFileClip

import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

# Load audio
y, sr = librosa.load("Steffner, Allan Biggs - Toca Toca [Tech House].mp3", sr=None)
duration = librosa.get_duration(y=y, sr=sr)

# Create folder for frames
os.makedirs("frames_waveform", exist_ok=True)

# Parameters
fps = 30
samples_per_frame = int(len(y) / (duration * fps))

for i in range(0, len(y), samples_per_frame):
    frame = y[i:i + samples_per_frame * 50]  # show a short window of signal
    plt.figure(figsize=(8, 4))
    plt.plot(frame, color='cyan')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"frames_waveform/frame_{i:06d}.png")
    plt.close()

print("✅ Waveform frames generated!")

# Create video from frames and add audio
audio = AudioFileClip("Steffner, Allan Biggs - Toca Toca [Tech House].mp3")
clip = ImageSequenceClip("frames_waveform", fps=30)
clip = clip.with_audio(audio)  # Add audio to the clip after creation
clip.write_videofile("output_video.mp4", codec="libx264", audio_codec="aac")

print("Video file 'output_video.mp4' created successfully.")