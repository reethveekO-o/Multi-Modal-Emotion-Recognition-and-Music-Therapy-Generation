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
import re
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

# === Enhanced Intensity Calculation (Based on Actual Audio) ===
def calculate_audio_intensity(audio_path):
    """Calculate actual audio intensity using RMS energy"""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    rms_energy = librosa.feature.rms(y=y)[0]
    mean_rms = np.mean(rms_energy)
    
    # Calibrated thresholds for intensity classification
    if mean_rms < 0.015:
        return 'low', mean_rms
    elif mean_rms < 0.08:
        return 'mid', mean_rms
    else:
        return 'high', mean_rms

# === Threatening/Violent Text Detection ===
def detect_threatening_content(text):
    """Detect threatening or violent language patterns"""
    threatening_patterns = [
        r'\b(kill|murder|hurt|harm|attack|destroy|beat|punch|hit)\b',
        r'\b(die|death|dead|violence|violent)\b',
        r'\b(hate|revenge|payback|suffer)\b'
    ]
    
    text_lower = text.lower()
    for pattern in threatening_patterns:
        if re.search(pattern, text_lower):
            return True
    return False

# === Positive Language Detection ===
def detect_positive_language(text):
    """Detect positive emotional language"""
    positive_patterns = [
        r'\b(love|like|happy|joy|wonderful|great|amazing|fantastic)\b',
        r'\b(good|nice|beautiful|awesome|excellent|perfect)\b',
        r'\b(thanks|thank you|grateful|appreciate)\b'
    ]
    
    text_lower = text.lower()
    for pattern in positive_patterns:
        if re.search(pattern, text_lower):
            return True
    return False

# === Disagreement Detection ===
def detect_disagreement(text_emotion, audio_emotion, text_conf, audio_conf):
    """Detect when text and audio emotions conflict"""
    
    # Define emotion categories for conflict detection
    positive_emotions = ['joy', 'love', 'happy', 'calm']
    negative_emotions = ['sadness', 'anger', 'fear', 'sad', 'angry', 'fearful', 'disgust']
    neutral_emotions = ['neutral', 'surprise', 'surprised']
    
    # Categorize emotions
    text_category = 'positive' if text_emotion in positive_emotions else \
                   'negative' if text_emotion in negative_emotions else 'neutral'
    audio_category = 'positive' if audio_emotion in positive_emotions else \
                    'negative' if audio_emotion in negative_emotions else 'neutral'
    
    # Calculate disagreement score
    disagreement_score = 0
    
    if text_category != audio_category and text_category != 'neutral' and audio_category != 'neutral':
        disagreement_score = abs(text_conf - audio_conf) / 100.0
    
    # High confidence threshold for reliable disagreement detection
    min_confidence = 60.0
    is_disagreement = (disagreement_score > 0.2 and 
                      text_conf > min_confidence and 
                      audio_conf > min_confidence)
    
    return is_disagreement, disagreement_score, text_category, audio_category

# === Smart Emotion Mapping ===
def smart_emotion_mapping(text_emotion, audio_emotion, text_conf, audio_conf, 
                         intensity_level, intensity_value, transcription):
    """Advanced emotion mapping with context awareness"""
    
    # Detect content patterns
    is_threatening = detect_threatening_content(transcription)
    is_positive_lang = detect_positive_language(transcription)
    
    # Detect disagreement
    is_disagreement, disagreement_score, text_cat, audio_cat = detect_disagreement(
        text_emotion, audio_emotion, text_conf, audio_conf
    )
    
    # Context-specific resolution strategies
    context_info = {
        'disagreement_detected': is_disagreement,
        'disagreement_score': disagreement_score,
        'threatening_content': is_threatening,
        'positive_language': is_positive_lang,
        'intensity_level': intensity_level,
        'intensity_value': intensity_value
    }
    
    # === Special Case 1: Threatening text with positive/happy audio ===
    if is_threatening and audio_emotion in ['happy', 'joy', 'calm'] and audio_conf > 50:
        if intensity_level in ['low', 'mid']:
            return 'Joking/Playful', 85, context_info
        else:
            return 'Sarcastic/Mocking', 75, context_info
    
    # === Special Case 2: Positive text with negative audio ===
    if is_positive_lang and audio_emotion in ['angry', 'sad', 'disgust'] and audio_conf > 50:
        if intensity_level == 'high':
            return 'Sarcasm/Irony', 80, context_info
        else:
            return 'Conflicted/Mixed', 70, context_info
    
    # === Special Case 3: Neutral text with strong emotional audio ===
    if text_emotion in ['neutral', 'surprise'] and audio_emotion in ['angry', 'sad', 'happy'] and audio_conf > 60:
        # Trust the audio emotion for neutral content
        emotion_mapping = {
            'angry': 'Frustrated',
            'sad': 'Disappointed', 
            'happy': 'Pleased',
            'fearful': 'Nervous',
            'disgust': 'Disgusted'
        }
        mapped_emotion = emotion_mapping.get(audio_emotion, audio_emotion.title())
        return f"{mapped_emotion} (Audio-driven)", audio_conf, context_info
    
    # === Confidence-based weighting ===
    if not is_disagreement:
        # No significant disagreement - use confidence weighting
        if text_conf > audio_conf + 20:  # Text much more confident
            base_emotion = text_emotion
            confidence = text_conf
        elif audio_conf > text_conf + 20:  # Audio much more confident
            base_emotion = audio_emotion
            confidence = audio_conf
        else:  # Similar confidence - use intensity modulation
            base_emotion = audio_emotion if intensity_level == 'high' else text_emotion
            confidence = (text_conf + audio_conf) / 2
        
        # Apply intensity modulation
        intensity_modulated_emotions = {
            ('sadness', 'high'): 'Despair',
            ('sadness', 'mid'): 'Melancholy', 
            ('sadness', 'low'): 'Mild Sadness',
            ('joy', 'high'): 'Excitement',
            ('joy', 'mid'): 'Happiness',
            ('joy', 'low'): 'Contentment',
            ('anger', 'high'): 'Rage',
            ('anger', 'mid'): 'Anger',
            ('anger', 'low'): 'Annoyance',
            ('fear', 'high'): 'Panic',
            ('fear', 'mid'): 'Fear',
            ('fear', 'low'): 'Anxiety',
            ('love', 'high'): 'Passion',
            ('love', 'mid'): 'Love',
            ('love', 'low'): 'Warmth',
            ('surprise', 'high'): 'Shock',
            ('surprise', 'mid'): 'Surprise',
            ('surprise', 'low'): 'Curiosity',
            ('happy', 'high'): 'Excitement',
            ('happy', 'mid'): 'Happiness', 
            ('happy', 'low'): 'Contentment',
            ('angry', 'high'): 'Rage',
            ('angry', 'mid'): 'Anger',
            ('angry', 'low'): 'Irritation',
            ('sad', 'high'): 'Despair',
            ('sad', 'mid'): 'Sadness',
            ('sad', 'low'): 'Melancholy'
        }
        
        final_emotion = intensity_modulated_emotions.get((base_emotion, intensity_level), 
                                                        f"{base_emotion.title()} ({intensity_level} intensity)")
        return final_emotion, confidence, context_info
    
    else:
        # Significant disagreement detected
        return f"Mixed/Conflicted ({text_cat} text, {audio_cat} audio)", 60, context_info

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

# === Enhanced Main Analysis Function ===
def net_emotion_summary():
    path = record_audio()
    path = preprocess_audio(path)
    segments, _ = whisper_model.transcribe(path)
    transcription = ' '.join(segment.text for segment in segments)
    print("\n📜 Transcribed Text:", transcription)

    # Get predictions from both modalities
    text_em, text_conf = predict_text_emotion(transcription)
    audio_em, audio_conf = predict_audio_emotion()
    
    # Calculate actual audio intensity
    intensity_level, intensity_value = calculate_audio_intensity(PROCESSED_FILENAME)
    
    # Apply smart emotion mapping
    final_emotion, final_confidence, context = smart_emotion_mapping(
        text_em, audio_em, text_conf, audio_conf, 
        intensity_level, intensity_value, transcription
    )

    # Display results
    print(f"\n🧠 Text Emotion: {text_em} ({text_conf:.2f}%)")
    print(f"🎧 Audio Emotion: {audio_em} ({audio_conf:.2f}%)")
    print(f"🔊 Audio Intensity: {intensity_level} (RMS: {intensity_value:.4f})")
    print(f"🎯 Final Emotion: {final_emotion} ({final_confidence:.2f}%)")
    
    # Display context information
    if context['disagreement_detected']:
        print(f"⚠️  Disagreement detected (score: {context['disagreement_score']:.2f})")
    if context['threatening_content']:
        print("🚨 Threatening content detected")
    if context['positive_language']:
        print("😊 Positive language detected")

# === Main Loop ===
if __name__ == '__main__':
    while True:
        print("\nType 'rec' to start or 'exit' to quit.")
        cmd = input(">>> ").strip().lower()
        if cmd == 'exit':
            break
        elif cmd == 'rec':
            net_emotion_summary()
