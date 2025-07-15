import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables all GPUs for TensorFlow and Keras
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
from emotion_final import run_pipeline as emotion_check

# Path to your trained video emotion model
VIDEO_MODEL_PATH = "models/video_model.h5"

# Emotions (should match your model's softmax class order)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion to stress mapping
STRESS_MAP = {
    'Angry': 0.93, 'Fear': 0.97, 'Sad': 0.92, 'Neutral': 0.38,
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

from collections import defaultdict
import numpy as np

def label_inertia(labels):
    """Returns +1 if no changes, -1 if alternates every frame, else in between."""
    if len(labels) < 2:
        return 0.0
    changes = sum(a != b for a, b in zip(labels[:-1], labels[1:]))
    return 1.0 - 2.0 * changes / (len(labels) - 1)

def compute_stress(predictions, w_var=0.1):
    """
    predictions : list[(frame_idx, emotion_label, confidence, class_idx)]
    returns      : final_stress, avg_stress, variability, inertia
    """

    # ---------- base stress & variability ----------
    emotions     = [e for _, e, _, _ in predictions]
    confidences  = [c for _, _, c, _ in predictions]
    stress_vals  = [STRESS_MAP[e] * c for e, c in zip(emotions, confidences)]

    avg_stress   = float(np.mean(stress_vals))
    variability  = float(np.std(stress_vals))

    # ---------- inertia (pair‑wise label correlation) ----------
    indices = [idx for _, _, _, idx in predictions]
    if len(indices) > 1:
        inertia = label_inertia(indices)
    else:
        inertia = 0.0

    # ---------- find the dominant emotion track ----------
    cum_conf = defaultdict(float)
    for e, c in zip(emotions, confidences):
        cum_conf[e] += c
    dominant_emotion = max(cum_conf, key=cum_conf.get)

    # ---------- inertia contribution as per your rule ----------
    if inertia < 0:                          # volatile sequence → add as is
        inertia_effect = -inertia
    else:                                    # stable sequence → scale by emotion tier
        if dominant_emotion in ("Happy", "Surprise"):
            inertia_effect = -0.05 * inertia
        elif dominant_emotion == "Neutral":
            inertia_effect =  0.10 * inertia
        elif dominant_emotion == "Sad":
            inertia_effect =  0.20 * inertia
        else:  # Angry, Fear, Disgust
            inertia_effect =  0.30 * inertia

    # ---------- final stress ----------
    final_stress = avg_stress + w_var * variability + inertia_effect
    final_stress = float(np.clip(final_stress, 0.0, 1.0))

    return final_stress, avg_stress, variability, inertia

def analyze_video_stress(video_path, final_emotion=None):
    model = load_model(VIDEO_MODEL_PATH, compile=False)
    frames, frame_nums, raw_frames = load_video_frames(video_path, num_frames=30)
    predictions = predict_video_emotions(frames, frame_nums, model)

    final_stress, base, var, inertia = compute_stress(predictions)
        # === NEW: obtain final emotion from emotion check and adjust stress ===
    if final_emotion is not None:
        NEGATIVE_EMOTIONS = {
            "anger", "angry", "rage", "irritation",
            "sadness", "sad", "despair", "melancholy",
            "fear", "panic", "anxiety",
            "disgust",
            "shock"   
        }
        STRESS_INCREMENT = {
            # Sadness cluster
            "sad": 0.3,
            "sadness": 0.3,
            "despair": 0.5,
            "melancholy": 0.2,

            # Anger cluster
            "angry": 0.4,
            "anger": 0.4,
            "rage": 0.5,
            "irritation": 0.2,

            # Fear cluster
            "fear": 0.4,
            "panic": 0.5,
            "anxiety": 0.5,   # high due to persistent stress

            # Disgust
            "disgust": 0.5,

            # Shock
            "shock": 0.4
        }


        final_emotion_lower = final_emotion.lower()
        if final_emotion_lower in NEGATIVE_EMOTIONS:
            increment = STRESS_INCREMENT.get(final_emotion_lower, 0.05)
            final_stress = float(np.clip(final_stress + increment, 0.0, 1.0))
            print(f"⚡ Final stress increased by {increment:.2f} due to detected emotion: {final_emotion}")
        else:
            print(f"✅ No negative emotion detected for adjustment. Detected emotion: {final_emotion}")
    else:
        print("⚠️ No emotion detected. Skipping emotion-based stress adjustment.")



    print("\n📈 Temporal Features:")
    print(f"- Avg. Emotion-based Stress: {base:.2f}")
    print(f"- Emotion Variability: {var:.2f}")
    print(f"- Emotion Inertia: {inertia:.2f}")
    print(f"\n🔍 Final Stress Score (0–1): {final_stress:.2f}")

    return final_stress

# Example usage
if __name__ == "__main__":
    video_path = r"C:\Users\vinit\OneDrive\Desktop\College\Research June-July\Multi-Modal-Emotion-Recognition-and-Music-Therapy-Generation\RecordedSession\output_video.avi"
    analyze_video_stress(video_path)
