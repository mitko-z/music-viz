import librosa

import librosa
import matplotlib.pyplot as plt
import os
import common_functions as cf

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

cf.create_video(
    music_file="Steffner, Allan Biggs - Toca Toca [Tech House].mp3",
    frames_folder="frames_waveform",
    output_file="output_video.mp4",
    fps=fps
)
