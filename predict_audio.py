import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import noisereduce as nr
import tensorflow as tf
import keyboard
import matplotlib.pyplot as plt
import threading
import time
import os

# === Configuration ===
SAMPLE_RATE = 16000
RAW_FILENAME = "live_audio.wav"
PROCESSED_FILENAME = "live_audio_preprocessed.wav"
MODEL_PATH = r"C:\Users\vinit\OneDrive\Desktop\College\Research June-July\Multi-Modal-Emotion-Recognition-and-Music-Therapy-Generation\models\audio_model.h5"
MAX_LENGTH = 126
FEATURE_DIM = 180

# === Emotion Mapping ===
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# === Focal Loss ===
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        p_t = y_true_one_hot * y_pred + (1 - y_true_one_hot) * (1 - y_pred)
        focal_weight = self.alpha * tf.pow((1 - p_t), self.gamma)
        focal_loss = focal_weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))

# === Load model ===
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'FocalLoss': FocalLoss})
print("✅ Model loaded successfully")

# === Record until SPACE ===
def record_audio(filename=RAW_FILENAME):
    print("Type 'rec' to begin or 'exit' to quit...")
    while True:
        user_input = input(">>> ").lower().strip()
        if user_input == "exit":
            print("👋 Exiting program.")
            exit(0)
        elif user_input == "rec":
            break

    print("🎙️ Recording... Press SPACE to stop.")
    recording = []
    stop_event = threading.Event()

    def space_listener():
        keyboard.wait('space')
        stop_event.set()

    def audio_callback(indata, frames, time_info, status):
        if stop_event.is_set():
            raise sd.CallbackAbort
        recording.append(indata.copy())

    threading.Thread(target=space_listener, daemon=True).start()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
        try:
            while not stop_event.is_set():
                time.sleep(0.1)
        except sd.CallbackAbort:
            pass

    audio = np.concatenate(recording).flatten()
    sf.write(filename, audio, SAMPLE_RATE)
    print(f"✅ Saved raw audio to {filename}")
    return filename

# === Preprocess and Save ===
def preprocess_audio(input_file, output_file):
    y, sr = librosa.load(input_file, sr=SAMPLE_RATE)
    y = librosa.util.normalize(y)

    if np.mean(np.abs(y)) > 0.01:
        y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.15)

    sf.write(output_file, y, sr)
    print(f"✅ Saved preprocessed audio to {output_file}")
    return output_file

# === Extract Features + Save MFCC Plot ===
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    y = librosa.util.normalize(y)

    if np.mean(np.abs(y)) > 0.01:
        y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.15)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)

    stacked = np.vstack([mfcc, mel_db, chroma]).T

    # === Save MFCC for visual analysis ===
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title("MFCC Structure")
    plt.tight_layout()
    plt.savefig("mfcc_structure.png")
    print("🎼 Saved MFCC structure to mfcc_structure.png")

    if stacked.shape[0] < MAX_LENGTH:
        pad = MAX_LENGTH - stacked.shape[0]
        stacked = np.pad(stacked, ((0, pad), (0, 0)))
    else:
        stacked = stacked[:MAX_LENGTH, :]

    return stacked

# === Predict Emotion ===
def predict_emotion(filepath):
    features = extract_features(filepath)
    input_tensor = np.expand_dims(features, axis=0)
    probs = model.predict(input_tensor, verbose=0)[0]

    pred_index = np.argmax(probs)
    confidence = probs[pred_index]
    emotion_label = emotion_map.get(f"{pred_index+1:02d}", f"Unknown ({pred_index})")

    print("\n🎯 Predicted Emotion:", emotion_label)
    print("📊 Confidence:", round(confidence * 100, 2), "%")

    labels = [emotion_map.get(f"{i+1:02d}", str(i)) for i in range(len(probs))]

    plt.figure(figsize=(10, 4))
    bars = plt.bar(labels, probs, color='skyblue')
    bars[pred_index].set_color('red')
    for i, prob in enumerate(probs):
        plt.text(i, prob + 0.01, f'{labels[i]}: {prob:.2f}', ha='center', fontsize=9)
    plt.title("Emotion Prediction Probabilities")
    plt.ylabel("Probability")
    plt.xlabel("Emotion")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("emotion_chart.png")
    print("📈 Saved emotion chart to emotion_chart.png")

# === Run Everything ===
if __name__ == "__main__":
    while True:
        raw_path = record_audio()
        processed_path = preprocess_audio(raw_path, PROCESSED_FILENAME)
        predict_emotion(processed_path)
