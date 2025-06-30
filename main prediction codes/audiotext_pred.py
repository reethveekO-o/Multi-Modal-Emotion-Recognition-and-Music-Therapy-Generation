"""
Enhanced Multimodal Emotion Recognition System - Text Prioritized Version
Fixes audio dominance issue by giving proper weight to high-confidence text predictions
"""
import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import noisereduce as nr
import tensorflow as tf
import keyboard
import torch
import re
import time
import threading
import os
import pickle
from scipy.stats import entropy
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
from faster_whisper import WhisperModel
import parselmouth
from parselmouth.praat import call
from xgboost import XGBClassifier

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ====================== CONSTANTS ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_FOLDER = os.path.join(BASE_DIR, "RecordedSession")
os.makedirs(SESSION_FOLDER, exist_ok=True)
RAW_FILENAME = os.path.join(SESSION_FOLDER, "output_audio.wav")
PROC_FILENAME = RAW_FILENAME  # Since preprocessing replaces in-place for consistency
SAMPLE_RATE = 16000
AUDIO_MODEL_PATH = "models/audio_model.h5"
TEXT_MODEL_PATH = "models/text_model"
AUTH_MODEL_PATH = "models/authenticity_xgb.pkl"
MAX_LENGTH = 126

# Emotion labels
TEXT_EMOTIONS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
AUDIO_EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Speaker-specific intensity history
speaker_history = []
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, help="Path to audio file for analysis")
args = parser.parse_args()

# ====================== HELPER FUNCTIONS ======================
def extract_voice_quality(audio_path):
    """Extract jitter, shimmer, HNR, and attack time using Parselmouth"""
    try:
        snd = parselmouth.Sound(audio_path)
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        
        # Extract acoustic features
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        hnr = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr_mean = call(hnr, "Get mean", 0, 0)
        
        # Calculate attack time
        squared = np.square(snd.values[0])
        diff = np.diff(squared)
        attack_idx = np.argmax(diff) if len(diff) > 0 else 0
        attack_time = attack_idx / snd.sampling_frequency
        
        return np.array([jitter, shimmer, hnr_mean, attack_time], dtype=np.float32)
    except:
        # Return default values if extraction fails
        return np.array([0.01, 0.05, 15.0, 0.1], dtype=np.float32)

def load_or_create_auth_model():
    """Load or create authenticity classifier"""
    if os.path.exists(AUTH_MODEL_PATH):
        try:
            with open(AUTH_MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except:
            pass
    
    # Create placeholder model for first run
    class DummyModel:
        def predict_proba(self, X):
            return np.array([[0.5, 0.5]])
    
    return DummyModel()

def calculate_confidence_entropy(probs):
    """Calculate confidence based on entropy"""
    try:
        return 1.0 - entropy(probs) / np.log(len(probs))
    except:
        return 0.5

# ====================== INTENSITY CALCULATION ======================
def calculate_adaptive_intensity(rms_value):
    """Calculate intensity level with speaker adaptation"""
    speaker_history.append(rms_value)
    
    # Use fixed thresholds until we have enough history
    if len(speaker_history) < 5:
        if rms_value < 0.015: return 'low'
        if rms_value < 0.08: return 'mid'
        return 'high'
    
    # Calculate percentiles based on speaker history
    low_thresh = np.percentile(speaker_history, 33)
    mid_thresh = np.percentile(speaker_history, 66)
    
    if rms_value < low_thresh:
        return 'low'
    elif rms_value < mid_thresh:
        return 'mid'
    else:
        return 'high'

def calculate_rms(audio_path):
    """Calculate RMS energy of audio"""
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    rms = librosa.feature.rms(y=y)
    return np.mean(rms)

# ====================== CONTENT ANALYSIS ======================
threatening_patterns = re.compile(
    r'\b(kill|murder|hurt|harm|attack|destroy|beat|punch|hit|die|death|dead|violence|violent|hate|revenge)\b', 
    re.IGNORECASE
)

positive_patterns = re.compile(
    r'\b(love|like|happy|joy|wonderful|great|amazing|fantastic|good|nice|beautiful|awesome|excellent|perfect|thanks|thank you|grateful|appreciate)\b', 
    re.IGNORECASE
)

negative_patterns = re.compile(
    r'\b(sad|tired|sick|deaths|disheartens|depressed|awful|terrible|horrible|hate|disgusted|angry|furious|scared|afraid|worried|anxious)\b', 
    re.IGNORECASE
)

def detect_threatening_content(text):
    return bool(threatening_patterns.search(text))

def detect_positive_language(text):
    return bool(positive_patterns.search(text))

def detect_negative_language(text):
    return bool(negative_patterns.search(text))

# ====================== MODEL LOADING ======================
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

# Load models
audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH, custom_objects={'FocalLoss': FocalLoss})
tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL_PATH)
text_model = BertForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
text_model.eval()
whisper_model = WhisperModel("base")
auth_model = load_or_create_auth_model()

# ====================== AUDIO PROCESSING ======================
def record_audio():
    """Record audio until SPACE is pressed"""
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

def preprocess_audio(audio_path):
    """Preprocess audio by reducing noise only (no normalization)"""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.15)
    sf.write(PROC_FILENAME, y, sr)
    return PROC_FILENAME

def extract_audio_features(audio_path):
    """Extract MFCC, Mel, and Chroma features without normalization"""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
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

# ====================== PREDICTION FUNCTIONS ======================
def predict_audio_emotion(audio_path):
    """Predict emotion from audio features"""
    features = extract_audio_features(audio_path)
    probs = audio_model.predict(features, verbose=0)[0]
    index = np.argmax(probs)
    label = AUDIO_EMOTIONS.get(f"{index+1:02d}", 'unknown')
    confidence = probs[index] * 100
    return label, confidence, probs

def predict_text_emotion(text):
    """Predict emotion from text"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze()
        idx = torch.argmax(probs).item()
        return TEXT_EMOTIONS[idx], probs[idx].item() * 100, probs.numpy()

# ====================== ENHANCED FUSION LOGIC ======================
def text_prioritized_fusion(text_emotion, audio_emotion, text_conf, audio_conf, 
                           text_probs, audio_probs, intensity, rms_value, transcription):
    """Text-prioritized fusion with improved logic"""
    
    # Calculate entropy-based confidence
    text_certainty = calculate_confidence_entropy(text_probs)
    audio_certainty = calculate_confidence_entropy(audio_probs)
    
    # Get authenticity score
    voice_features = extract_voice_quality(PROC_FILENAME)
    try:
        auth_score = auth_model.predict_proba([voice_features])[0][1]
    except:
        auth_score = 0.5
    
    # Content analysis
    is_threatening = detect_threatening_content(transcription)
    is_positive_lang = detect_positive_language(transcription)
    is_negative_lang = detect_negative_language(transcription)
    
    # Emotion categories
    positive_emotions = {'joy', 'love', 'happy', 'calm'}
    negative_emotions = {'sadness', 'anger', 'fear', 'sad', 'angry', 'fearful', 'disgust'}
    
    text_category = 'positive' if text_emotion in positive_emotions else \
                   'negative' if text_emotion in negative_emotions else 'neutral'
    audio_category = 'positive' if audio_emotion in positive_emotions else \
                    'negative' if audio_emotion in negative_emotions else 'neutral'
    
    # **KEY CHANGE: Much stronger text prioritization**
    
    # Rule 1: Very high text confidence (>95%) - TEXT DOMINATES
    if text_conf > 95.0:
        base_emotion = text_emotion
        confidence = text_conf
        reason = f"High text confidence ({text_conf:.1f}%)"
        
    # Rule 2: High text confidence (>85%) with supporting language patterns
    elif text_conf > 85.0 and (
        (text_category == 'negative' and is_negative_lang) or
        (text_category == 'positive' and is_positive_lang)
    ):
        base_emotion = text_emotion
        confidence = text_conf
        reason = f"Text confidence + language pattern match"
        
    # Rule 3: Threatening content handling
    elif is_threatening and audio_emotion in positive_emotions:
        if auth_score < 0.4:  # Low authenticity = sarcasm
            return 'Sarcasm/Irony', 80, auth_score, True, "Threatening + positive audio + low authenticity"
        elif auth_score > 0.6:  # High authenticity = real anger
            base_emotion = 'anger'
            confidence = max(text_conf, audio_conf)
            reason = "Authentic threatening content"
        else:  # Medium authenticity - trust text
            base_emotion = text_emotion
            confidence = text_conf
            reason = "Threatening content with medium authenticity"
            
    # Rule 4: Strong disagreement - check authenticity
    elif text_category != audio_category and text_category != 'neutral' and audio_category != 'neutral':
        confidence_diff = abs(text_conf - audio_conf)
        
        if text_conf > 80.0 and confidence_diff > 15:  # Text much more confident
            base_emotion = text_emotion
            confidence = text_conf
            reason = f"Text dominance (diff: {confidence_diff:.1f}%)"
        elif auth_score > 0.6 and audio_category == 'negative':  # Authentic negative audio
            base_emotion = audio_emotion
            confidence = audio_conf
            reason = "Authentic negative audio emotion"
        else:  # Default to text for disagreements
            base_emotion = text_emotion
            confidence = (text_conf * 0.7 + audio_conf * 0.3)
            reason = "Text-weighted disagreement resolution"
            
    # Rule 5: Agreement or weak disagreement - confidence weighting favoring text
    else:
        # Enhanced text weighting: 70% text, 30% audio
        text_weight = 0.7 + (text_certainty * 0.2)  # 70-90% text weight
        audio_weight = 1 - text_weight
        
        if text_conf > 70.0:  # Decent text confidence
            base_emotion = text_emotion
            confidence = text_conf * text_weight + audio_conf * audio_weight
            reason = f"Text-weighted fusion ({text_weight:.1f}/{audio_weight:.1f})"
        else:  # Low text confidence - use audio
            base_emotion = audio_emotion
            confidence = audio_conf
            reason = "Low text confidence - using audio"
    
    # Apply intensity modulation
    emotion_mapping = {
        ('anger', 'high'): 'Rage',
        ('anger', 'mid'): 'Anger', 
        ('anger', 'low'): 'Annoyance',
        ('sadness', 'high'): 'Despair',
        ('sadness', 'mid'): 'Sadness',
        ('sadness', 'low'): 'Melancholy',
        ('joy', 'high'): 'Excitement',
        ('joy', 'mid'): 'Happiness',
        ('joy', 'low'): 'Contentment',
        ('fear', 'high'): 'Panic',
        ('fear', 'mid'): 'Fear',
        ('fear', 'low'): 'Anxiety',
        ('love', 'high'): 'Passion',
        ('love', 'mid'): 'Love',
        ('love', 'low'): 'Warmth',
        ('surprise', 'high'): 'Shock',
        ('surprise', 'mid'): 'Surprise',
        ('surprise', 'low'): 'Curiosity',
        ('angry', 'high'): 'Rage',
        ('angry', 'mid'): 'Anger',
        ('angry', 'low'): 'Irritation',
        ('sad', 'high'): 'Despair',
        ('sad', 'mid'): 'Sadness',
        ('sad', 'low'): 'Melancholy',
        ('happy', 'high'): 'Excitement',
        ('happy', 'mid'): 'Happiness',
        ('happy', 'low'): 'Contentment'
    }
    
    # Only apply intensity modulation if confidence is reasonable
    if confidence > 60:
        final_emotion = emotion_mapping.get((base_emotion, intensity), f"{base_emotion.title()}")
    else:
        final_emotion = base_emotion.title()
    
    disagreement_detected = text_category != audio_category and text_category != 'neutral' and audio_category != 'neutral'
    
    return final_emotion, confidence, auth_score, disagreement_detected, reason

# ====================== MAIN ANALYSIS FUNCTION ======================
def analyze_emotion():
    """Main pipeline for emotion analysis"""
    # Record and preprocess audio
    if args.file:
        print(f"🔹 Using provided audio file: {args.file}")
        raw_path = args.file
    else:
        raw_path = record_audio()

    proc_path = preprocess_audio(raw_path)

    
    # Transcribe audio
    segments, _ = whisper_model.transcribe(proc_path)
    transcription = ' '.join(segment.text for segment in segments).strip() or "..."
    # Save transcript for pipeline consistency
    with open(os.path.join(SESSION_FOLDER, "final_output.txt"), "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"✅ Saved transcription to RecordedSession/final_output.txt")

    print("\n📜 Transcription:", transcription)
    
    # Get predictions
    text_em, text_conf, text_probs = predict_text_emotion(transcription)
    audio_em, audio_conf, audio_probs = predict_audio_emotion(proc_path)
    
    # Calculate intensity
    rms = calculate_rms(proc_path)
    intensity = calculate_adaptive_intensity(rms)
    
    # Apply text-prioritized fusion
    final_emotion, final_conf, auth_score, disagreement, reason = text_prioritized_fusion(
        text_em, audio_em, text_conf, audio_conf,
        text_probs, audio_probs, intensity, rms, transcription
    )
    
    # Display results
# === (imports and other code remain unchanged) ===

# Replace your final print block in `analyze_emotion()` with this:
    print(f"\n🧠 Text Emotion: {text_em} ({text_conf:.2f}%)")
    print(f"🎧 Audio Emotion: {audio_em} ({audio_conf:.2f}%)")
    print(f"🔊 Intensity: {intensity} (RMS: {rms:.4f})")
    print(f"🔍 Authenticity Score: {auth_score:.2f}")
    print(f"💡 Reasoning: {reason}")
    
    if disagreement:
        print("⚠️  Disagreement detected between modalities")
    if detect_threatening_content(transcription):
        print("🚨 Threatening content detected")
    if detect_positive_language(transcription):
        print("😊 Positive language detected")
    if detect_negative_language(transcription):
        print("😢 Negative language detected")
    
    # 👇 Clean final parseable result for pipeline
    print(f"Emotion: {final_emotion} ({final_conf:.2f}%)")


# ====================== MAIN LOOP ======================
if __name__ == '__main__':
    if args.file:
        print("===== Running Emotion Analysis on Provided File =====")
        analyze_emotion()
    else:
        print("===== Text-Prioritized Emotion Recognition System =====")
        print("Type 'rec' to start recording or 'exit' to quit")
        
        while True:
            command = input("\n>>> ").strip().lower()
            
            if command == 'exit':
                print("Exiting...")
                break
                
            elif command == 'rec':
                print("Starting emotion analysis...")
                try:
                    analyze_emotion()
                except Exception as e:
                    print(f"Error: {e}")
                    speaker_history.clear()
                    
            else:
                print("Invalid command. Type 'rec' or 'exit'")