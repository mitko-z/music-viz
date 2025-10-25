import librosa
import numpy as np
import cv2
import os
import common_functions as cf

# Load audio
audio_file_name = "Steffner, Allan Biggs - Toca Toca [Tech House].mp3"
y, sr = librosa.load(audio_file_name, sr=None)
hop_length = 1024
n_fft = 2048
S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
S_db = librosa.amplitude_to_db(S, ref=np.max)

frames_folder = "frames_circle_" + audio_file_name.split(".")[0]
os.makedirs(frames_folder, exist_ok=True)
num_bars = 120
height, width = 720, 1280
center = (width // 2, height // 2)
base_radius = 200
color = (255, 80, 255)  # pink-magenta

prev_values = np.zeros(num_bars)
decay = 0.85  # for smooth falloff

freqs = np.linspace(0, S_db.shape[0]-1, num_bars, dtype=int)

for i in range(0, S_db.shape[1], 2):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    raw_values = S_db[freqs, i]
    new_values = cf.normalize_and_smooth(raw_values)

    # Smooth motion decay (prevents flickering)
    current_values = np.maximum(new_values, prev_values * decay)
    prev_values = current_values

    # Draw bars
    for j, val in enumerate(current_values):
        angle = 2 * np.pi * j / num_bars
        x1 = int(center[0] + base_radius * np.cos(angle))
        y1 = int(center[1] + base_radius * np.sin(angle))
        x2 = int(center[0] + (base_radius + val) * np.cos(angle))
        y2 = int(center[1] + (base_radius + val) * np.sin(angle))
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    # Optional: dotted circle guide
    for a in np.linspace(0, 2*np.pi, num_bars*2):
        cx = int(center[0] + base_radius * np.cos(a))
        cy = int(center[1] + base_radius * np.sin(a))
        frame[cy, cx] = (80, 20, 80)

    cv2.imwrite(f"{frames_folder}/frame_{i:06d}.png", frame)

print("✅ Circular visualizer frames generated!")

cf.create_video(
    music_file=audio_file_name,
    frames_folder=frames_folder,
    output_file=audio_file_name.split(".")[0] + "_radial_bars_video.mp4",
    fps=30
)