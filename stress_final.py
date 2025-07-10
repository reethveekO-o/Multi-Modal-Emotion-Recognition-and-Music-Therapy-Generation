import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time

# Load your trained model
MODEL_PATH = "code/video_model.h5"  # Ensure the model is in your current directory
model = load_model(MODEL_PATH, compile=False)

# Emotion classes (must match model output)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion-to-stress mapping
STRESS_MAP = {
    'Angry': 0.93, 'Fear': 0.97, 'Sad': 0.92, 'Neutral': 0.39,
    'Happy': 0.1, 'Surprise': 0.35, 'Disgust': 0.99
}

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    norm = resized.astype('float32') / 255.0
    return norm

def predict_emotion(frame):
    input_tensor = frame[np.newaxis, ..., np.newaxis]
    prob = model.predict(input_tensor, verbose=0)[0]
    idx = np.argmax(prob)
    return EMOTION_LABELS[idx], prob[idx], idx, prob

def compute_stress(predictions):
    indices = [idx for _, _, idx, _ in predictions]
    confidences = [c for _, c, _, _ in predictions]
    emotions = [e for e, _, _, _ in predictions]

    stress_vals = [STRESS_MAP.get(e, 0.5) * c for e, c in zip(emotions, confidences)]
    avg_stress = np.mean(stress_vals)
    variability = np.std(indices)

    if len(indices) > 1:
        inertia = np.corrcoef(indices[:-1], indices[1:])[0, 1]
        if np.isnan(inertia):
            inertia = 0.0
    else:
        inertia = 0.0

    final_stress = avg_stress + 0.1 * variability - 0.05 * inertia
    return np.clip(final_stress, 0, 1), avg_stress, variability, inertia

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("📸 Capturing 30 frames from webcam for analysis...")
    predictions = []
    raw_frames = []

    while len(predictions) < 30:
        ret, frame = cap.read()
        if not ret:
            continue

        raw_frames.append(frame.copy())
        processed = preprocess_frame(frame)
        emotion, conf, idx, _ = predict_emotion(processed)

        predictions.append((emotion, conf, idx, frame))

        # Display with overlay
        cv2.putText(frame, f"{emotion} ({conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Emotion Detection", frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    final_stress, avg_stress, var, inertia = compute_stress(predictions)

    print("\n📊 Frame-by-frame predictions:")
    for i, (emotion, conf, idx, frame) in enumerate(predictions):
        print(f"Frame {i+1:02d}: {emotion} ({conf:.2f})")

    print("\n📈 Temporal Features:")
    print(f"- Avg. Emotion-based Stress: {avg_stress:.2f}")
    print(f"- Emotion Variability: {var:.2f}")
    print(f"- Emotion Inertia: {inertia:.2f}")
    print(f"\n🔍 Final Stress Score (0-1): {final_stress:.2f}")

if __name__ == "__main__":
    main()
