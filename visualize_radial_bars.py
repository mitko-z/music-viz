import librosa
import numpy as np
import cv2
import os
import common_functions as cf

data_folder = "data"

# Load audio
audio_file_name = "Steffner, Allan Biggs - Toca Toca [Tech House].mp3"
audio_file_name = os.path.join(data_folder, audio_file_name)
y, sr = librosa.load(audio_file_name, sr=None)
hop_length = 1024
n_fft = 2048
S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
S_db = librosa.amplitude_to_db(S, ref=np.max)

frames_folder = os.path.join(data_folder, "frames_circle_" + audio_file_name.split(".")[0])
os.makedirs(frames_folder, exist_ok=True)
num_bars = 120
height, width = 720, 1280
center = (width // 2, height // 2)
base_radius = 200
color = (255, 80, 255)  # pink-magenta


# === Load and prepare background  ===
background_image_file_name = "Gemini_Generated_Image_xo6y9cxo6y9cxo6y_expanded.jpg"
background_image_file_name = os.path.join(data_folder, background_image_file_name)
background = cv2.imread(background_image_file_name, cv2.IMREAD_UNCHANGED)
if background is None:
    raise FileNotFoundError(f"{background_image_file_name} not found!")

center = (background.shape[1] // 2, background.shape[0] // 2)

# === Generate frames with radial bars ===
prev_values = np.zeros(num_bars)
decay = 0.85  # for smooth falloff

freqs = np.linspace(0, S_db.shape[0] - 1, num_bars, dtype=int)

# Define separate centers
frame_center = (width // 2, height // 2)  # center of the output frame
bg_center = (background.shape[1] // 2, background.shape[0] // 2)  # center of the background

# --- Precompute vignette mask (static, same for all frames) ---
X = np.linspace(-1, 1, width)
Y = np.linspace(-1, 1, height)
xv, yv = np.meshgrid(X, Y)
radius = np.sqrt(xv**2 + yv**2)
vignette_mask = np.clip(1 - radius**1.8, 0, 1)  # adjust exponent for strength
vignette_mask = cv2.GaussianBlur(vignette_mask, (0, 0), sigmaX=120)
vignette_mask = vignette_mask[..., None]  # make it 3D for RGB multiplication

# Precompute onset strength or RMS
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
onset_env = librosa.util.normalize(onset_env)

# Interpolate so that each visual frame corresponds to one audio frame
onset_env_interp = np.interp(np.linspace(0, len(onset_env), S_db.shape[1]), np.arange(len(onset_env)), onset_env)

for i in range(0, S_db.shape[1], 2):
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # --- Background rotation (optional) ---
    speed = 0  # degrees per frame (set to e.g. 0.5 for slow rotation)
    angle = (i * speed) % 360
    rot_mat = cv2.getRotationMatrix2D(bg_center, angle, 1.0)
    rotated = cv2.warpAffine(
        background,
        rot_mat,
        (background.shape[1], background.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )
    # --- Beat-based bounce effect ---
    energy = onset_env_interp[i]
    scale_factor = 1.0 + 0.1 * energy  # 0.1 = max +10% zoom
    rot_mat = cv2.getRotationMatrix2D(bg_center, 0, scale_factor)
    rotated = cv2.warpAffine(
        background,
        rot_mat,
        (background.shape[1], background.shape[0]),
        borderMode=cv2.BORDER_REFLECT
    )


    # --- Center the background on the video frame ---
    y1 = frame_center[1] - background.shape[0] // 2
    x1 = frame_center[0] - background.shape[1] // 2
    y2 = y1 + background.shape[0]
    x2 = x1 + background.shape[1]

    # Clip safely to frame bounds
    y1_frame = max(0, y1)
    x1_frame = max(0, x1)
    y2_frame = min(height, y2)
    x2_frame = min(width, x2)

    # Corresponding region in rotated image
    ry1 = y1_frame - y1
    rx1 = x1_frame - x1
    ry2 = ry1 + (y2_frame - y1_frame)
    rx2 = rx1 + (x2_frame - x1_frame)

    if (y2_frame > y1_frame) and (x2_frame > x1_frame):
        roi = frame[y1_frame:y2_frame, x1_frame:x2_frame]
        cropped = rotated[ry1:ry2, rx1:rx2]

        # If image has alpha channel, blend with transparency
        if cropped.shape[2] == 4:
            bgr = cropped[..., :3].astype(np.float32)
            alpha = (cropped[..., 3].astype(np.float32) / 255.0)[..., None]
            overlay = cv2.addWeighted(roi.astype(np.float32), 1.0, bgr, 0.8, 0)
            comp = (overlay * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            frame[y1_frame:y2_frame, x1_frame:x2_frame] = comp
        else:
            frame[y1_frame:y2_frame, x1_frame:x2_frame] = cv2.addWeighted(
                roi, 1.0, cropped, 0.8, 0
            )

    # --- Draw circular bars ---
    raw_values = S_db[freqs, i]
    new_values = cf.normalize_and_smooth(raw_values)

    # Smooth motion decay
    current_values = np.maximum(new_values, prev_values * decay)
    prev_values = current_values

    # Create glow layer (draw bars thicker and blurred)
    glow_layer = np.zeros_like(frame, dtype=np.uint8)
    for j, val in enumerate(current_values):
        angle = 2 * np.pi * j / num_bars
        x1 = int(frame_center[0] + base_radius * np.cos(angle))
        y1 = int(frame_center[1] + base_radius * np.sin(angle))
        x2 = int(frame_center[0] + (base_radius + val) * np.cos(angle))
        y2 = int(frame_center[1] + (base_radius + val) * np.sin(angle))
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    # --- Optional dotted circular guide ---
    for a in np.linspace(0, 2 * np.pi, num_bars * 2):
        cx = int(frame_center[0] + base_radius * np.cos(a))
        cy = int(frame_center[1] + base_radius * np.sin(a))
        if 0 <= cx < width and 0 <= cy < height:
            frame[cy, cx] = (80, 20, 80)

    cv2.imwrite(f"{frames_folder}/frame_{i:06d}.png", frame)

print("✅ Circular visualizer frames generated!")


cf.create_video(
    music_file=audio_file_name,
    frames_folder=frames_folder,
    output_file=audio_file_name.split(".")[0] + "_radial_bars_video.mp4",
    fps=30
)