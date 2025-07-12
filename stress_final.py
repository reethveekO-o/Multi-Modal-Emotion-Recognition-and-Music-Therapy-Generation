import os
import logging
import warnings
import sys
# Suppress TensorFlow, xFormers, and general warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("xformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from tqdm import tqdm

# Path to your trained video emotion model
VIDEO_MODEL_PATH = "models/video_model.h5"

# Emotions (should match your model's softmax class order)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion to stress mapping
STRESS_MAP = {
    'Angry': 0.93, 'Fear': 0.97, 'Sad': 0.92, 'Neutral': 0.39,
    'Happy': 0.1, 'Surprise': 0.35, 'Disgust': 0.99
}

# Load OpenCV Haar Cascade
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_video_frames(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    frame_nums = []
    raw_frames = []

    pbar = tqdm(total=len(indices), desc="Processing frames", ncols=80)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            pbar.update(1)
            continue
        raw_frames.append(frame.copy())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces):
            x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
            face = gray[y:y+h, x:x+w]
        else:
            face = gray  # fallback to full frame

        resized = cv2.resize(face, (48, 48))
        norm = resized.astype('float32') / 255.0
        frames.append(norm)
        frame_nums.append(int(idx))
        pbar.update(1)

    pbar.close()
    cap.release()
    return np.array(frames), frame_nums, raw_frames

def predict_video_emotions(frames, frame_nums, model):
    preds = []
    for f, frame_idx in zip(frames, frame_nums):
        input_tensor = f[np.newaxis, ..., np.newaxis]
        prob = model.predict(input_tensor, verbose=0)[0]
        idx = np.argmax(prob)
        emotion = EMOTION_LABELS[idx]
        confidence = prob[idx]
        preds.append((frame_idx, emotion, confidence, idx))
    return preds

def compute_stress(predictions):
    indices = [idx for _, _, _, idx in predictions]
    confidences = [c for _, _, c, _ in predictions]
    emotions = [e for _, e, _, _ in predictions]

    stress_vals = [STRESS_MAP.get(e, 0.5) * c for e, c in zip(emotions, confidences)]
    avg_stress = np.mean(stress_vals)
    variability = np.std(stress_vals)

    if len(indices) > 1:
        inertia = np.corrcoef(indices[:-1], indices[1:])[0, 1]
        if np.isnan(inertia):
            inertia = 0.0
    else:
        inertia = 0.0

    final_stress = avg_stress + 0.1 * variability - 0.05 * inertia
    final_stress = np.clip(final_stress, 0, 1)

    return final_stress, avg_stress, variability, inertia

def analyze_video_stress(video_path):
    model = load_model(VIDEO_MODEL_PATH, compile=False)
    frames, frame_nums, raw_frames = load_video_frames(video_path, num_frames=30)
    predictions = predict_video_emotions(frames, frame_nums, model)

    final_stress, base, var, inertia = compute_stress(predictions)

    print("\n📊 Frame-by-frame predictions:")
    for i, (frame_idx, emotion, conf, _) in enumerate(predictions):
        print(f"Frame {i+1:02d} [Index {frame_idx:04d}]: {emotion} ({conf:.2f})")

        frame_disp = raw_frames[i].copy()
        cv2.putText(frame_disp, f"{emotion} ({conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        output_dir = "annotated_frames"
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f"frame_{i+1:02d}.jpg"), frame_disp)

    print("\n📈 Temporal Features:")
    print(f"- Avg. Emotion-based Stress: {base:.2f}")
    print(f"- Emotion Variability: {var:.2f}")
    print(f"- Emotion Inertia: {inertia:.2f}")
    print(f"\n🔍 Final Stress Score (0–1): {final_stress:.2f}")

    return final_stress

# Example usage
if __name__ == "__main__":
    video_path = r"C:\Users\rithvik\OneDrive\Documents\GitHub\Multi-Modal-Emotion-Recognition-and-Music-Therapy-Generation\RecordedSession\output_video.avi"
    analyze_video_stress(video_path)
