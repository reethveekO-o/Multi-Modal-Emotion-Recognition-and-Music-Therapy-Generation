import numpy as np
import cv2
from tensorflow.keras.models import load_model
from collections import Counter
import time
import os
from tqdm import tqdm

# ======== CONFIG ========
session_folder = "RecordedSession"
os.makedirs(session_folder, exist_ok=True)

video_file = os.path.join(session_folder, "final_output.mp4")
fallback_video_file = os.path.join(session_folder, "recorded_emotion_video.avi")
model_path = r"C:\Users\rithvik\OneDrive\Documents\GitHub\Multi-Modal-Emotion-Recognition-and-Music-Therapy-Generation\models\video_model.h5"
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ======== MODEL ========
model = load_model(model_path)

# ======== FACE DETECTOR ========
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ======== FRAME PREPROCESSING ========
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face = gray[y:y+h, x:x+w]
    resized = cv2.resize(face, (48, 48))
    normalized = resized.astype('float32') / 255.0
    return normalized.reshape(1, 48, 48, 1)

# ======== PREDICT EMOTION FROM VIDEO ========
def predict_from_video(video_path, num_samples=30, aggregate='softmax'):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, frame_count - 1, num_samples, dtype=int)

    softmax_outputs = []
    pred_labels = []

    idx = 0
    pbar = tqdm(total=num_samples, desc="Processing frames", ncols=80)

    while cap.isOpened() and idx < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if current_frame in sample_indices:
            face_input = preprocess_frame(frame)
            if face_input is not None:
                pred = model.predict(face_input, verbose=0)[0]
                softmax_outputs.append(pred)
                pred_labels.append(np.argmax(pred))
            idx += 1
            pbar.update(1)

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()

    if not softmax_outputs:
        return "No faces detected", 0.0, None

    if aggregate == 'softmax':
        avg_softmax = np.mean(softmax_outputs, axis=0)
        final_emotion = emotion_labels[np.argmax(avg_softmax)]
        confidence = np.max(avg_softmax) * 100
        return final_emotion, confidence, avg_softmax
    else:
        majority = Counter(pred_labels).most_common(1)[0]
        final_emotion = emotion_labels[majority[0]]
        confidence = (majority[1] / len(pred_labels)) * 100
        return final_emotion, confidence, None

# ======== FALLBACK RECORDING ========
def record_fallback_video(duration=5, output_filename=fallback_video_file):
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    print("🎥 Recording fallback video for analysis. Press SPACE to stop early.")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        cv2.imshow("Recording (Press SPACE to stop)", frame)

        if (time.time() - start_time) >= duration:
            print("✅ Recording completed by duration.")
            break
        if cv2.waitKey(1) & 0xFF == ord(' '):
            print("✅ Recording stopped by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_filename

# ======== MAIN EXECUTION ========
if __name__ == "__main__":
    print("===== Video Emotion Recognition (Modal Extraction Consistent) =====")
    try:
        if os.path.exists(video_file):
            print(f"✅ Using modal extraction video: {video_file}")
            target_video = video_file
        else:
            print("⚠️ Modal extraction video not found. Recording fallback...")
            target_video = record_fallback_video(duration=5)

        emotion, conf, avg_softmax = predict_from_video(target_video, num_samples=30)

        # Clean, pipeline-friendly output
        print(f"Emotion: {emotion} ({conf:.2f}%)")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully.")
