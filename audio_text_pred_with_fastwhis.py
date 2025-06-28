import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import noisereduce as nr
import tensorflow as tf
import keyboard
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import threading
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# === Faster Whisper ===
from faster_whisper import WhisperModel

# === Configuration ===
SAMPLE_RATE = 16000
RAW_FILENAME = "live_audio.wav"
PROCESSED_FILENAME = "live_audio_preprocessed.wav"
AUDIO_MODEL_PATH = r"models/audio_model.h5"
TEXT_MODEL_PATH = r"models/text_model"
MAX_LENGTH = 126

# === Emotion Labels ===
text_emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
audio_emotions = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# === Intensity Mapping ===
intensity_mapping = {
    'sad': 'low', 'joy': 'high', 'love': 'low', 'anger': 'high',
    'fear': 'high', 'surprise': 'high', 'neutral': 'low',
    'calm': 'low', 'happy': 'high', 'disgust': 'high',
    'fearful': 'high', 'surprised': 'high'
}

# === Net Emotion Combinations ===
net_emotion_mapping = {
    ('sadness', 'high'): 'Despair',
    ('sadness', 'low'): 'Melancholy',
    ('joy', 'high'): 'Excitement',
    ('joy', 'low'): 'Contentment',
    ('anger', 'high'): 'Rage',
    ('anger', 'low'): 'Annoyance',
    ('fear', 'high'): 'Panic',
    ('fear', 'low'): 'Anxiety',
    ('love', 'high'): 'Passion',
    ('love', 'low'): 'Warmth',
    ('surprise', 'high'): 'Shock',
    ('surprise', 'low'): 'Curiosity',
    ('neutral', 'low'): 'Neutral'
}

# === Audio Model (with Focal Loss) ===
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

model_audio = tf.keras.models.load_model(AUDIO_MODEL_PATH, custom_objects={'FocalLoss': FocalLoss})

# === Load Text Model ===
tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL_PATH)
model_text = BertForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
model_text.eval()

# === Load Whisper Model ===
whisper_model = WhisperModel("base")

# === Record Audio ===
def record_audio():
    print("🎙️ Recording... Press SPACE to stop.")
    recording = []
    stop_event = threading.Event()

    def space_listener():
        keyboard.wait('space')
        stop_event.set()

    threading.Thread(target=space_listener, daemon=True).start()

    def callback(indata, frames, time_info, status):
        if stop_event.is_set():
            raise sd.CallbackAbort
        recording.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        try:
            while not stop_event.is_set():
                time.sleep(0.1)
        except sd.CallbackAbort:
            pass

    audio = np.concatenate(recording).flatten()
    sf.write(RAW_FILENAME, audio, SAMPLE_RATE)
    print(f"✅ Saved raw audio to {RAW_FILENAME}")
    return RAW_FILENAME

# === Preprocess Audio ===
def preprocess_audio(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    y = librosa.util.normalize(y)
    y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.15)
    sf.write(PROCESSED_FILENAME, y, sr)
    return PROCESSED_FILENAME

# === Extract Features ===
def extract_features(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    stacked = np.vstack([mfcc, mel_db, chroma]).T
    if stacked.shape[0] < MAX_LENGTH:
        pad = MAX_LENGTH - stacked.shape[0]
        stacked = np.pad(stacked, ((0, pad), (0, 0)))
    else:
        stacked = stacked[:MAX_LENGTH, :]
    return np.expand_dims(stacked, axis=0)

# === Predict Audio Emotion ===
def predict_audio_emotion():
    features = extract_features(PROCESSED_FILENAME)
    probs = model_audio.predict(features, verbose=0)[0]
    index = np.argmax(probs)
    label = audio_emotions.get(f"{index+1:02d}", 'unknown')
    confidence = probs[index] * 100
    return label, confidence

# === Predict Text Emotion ===
def predict_text_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model_text(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze()
        idx = torch.argmax(probs).item()
        return text_emotions[idx], probs[idx].item() * 100

# === Main Analysis Function ===
def net_emotion_summary():
    path = record_audio()
    path = preprocess_audio(path)
    segments, _ = whisper_model.transcribe(path)
    transcription = ' '.join(segment.text for segment in segments)
    print("\n📜 Transcribed Text:", transcription)

    text_em, text_conf = predict_text_emotion(transcription)
    audio_em, audio_conf = predict_audio_emotion()
    intensity = intensity_mapping.get(audio_em, 'low')
    net_em = net_emotion_mapping.get((text_em, intensity), 'Mixed State')

    print(f"\n🧠 Text Emotion: {text_em} ({text_conf:.2f}%)")
    print(f"🎧 Audio Emotion: {audio_em} ({audio_conf:.2f}%) → Intensity: {intensity}")
    print(f"🎯 Net Emotion Interpretation: {net_em}")

# === Main Loop ===
if __name__ == '__main__':
    while True:
        print("\nType 'rec' to start or 'exit' to quit.")
        cmd = input(">>> ").strip().lower()
        if cmd == 'exit':
            break
        elif cmd == 'rec':
            net_emotion_summary()
