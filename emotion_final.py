# =================== FULL MULTIMODAL EMOTION RECOGNITION PIPELINE (AUTH REMOVED) ===================

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
import subprocess
import cv2
from scipy.stats import entropy
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
from faster_whisper import WhisperModel
from tensorflow.keras.models import load_model
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

# =================== CONFIG ===================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_FOLDER = os.path.join(BASE_DIR, "RecordedSession")
os.makedirs(SESSION_FOLDER, exist_ok=True)

RAW_AUDIO_FILE = os.path.join(SESSION_FOLDER, "output_audio.wav")
PROC_AUDIO_FILE = os.path.join(SESSION_FOLDER, "processed_audio.wav")
VIDEO_FILE = os.path.join(SESSION_FOLDER, "output_video.avi")
FINAL_OUTPUT_FILE = os.path.join(SESSION_FOLDER, "final_output.mp4")
TRANSCRIPT_FILE = os.path.join(SESSION_FOLDER, "final_output.txt")

SAMPLE_RATE = 16000
FPS = 20
FRAME_SIZE = (640, 480)
MAX_LENGTH = 126

AUDIO_MODEL_PATH = "models/audio_model.h5"
TEXT_MODEL_PATH = "models/text_model"
VIDEO_MODEL_PATH = "models/video_model.h5"

TEXT_EMOTIONS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
AUDIO_EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
VIDEO_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

threatening_patterns = re.compile(r'\\b(kill|murder|hurt|harm|attack|destroy|beat|punch|hit|die|death|dead|violence|violent|hate|revenge)\\b', re.IGNORECASE)
positive_patterns = re.compile(r'\\b(love|like|happy|joy|wonderful|great|amazing|fantastic|good|nice|beautiful|awesome|excellent|perfect|thanks|thank you|grateful|appreciate)\\b', re.IGNORECASE)
negative_patterns = re.compile(r'\\b(sad|tired|sick|deaths|disheartens|depressed|awful|terrible|horrible|hate|disgusted|angry|furious|scared|afraid|worried|anxious)\\b', re.IGNORECASE)

emotion_mapping = {
    ('anger', 'high'): 'Rage', ('anger', 'mid'): 'Anger', ('anger', 'low'): 'Irritation',
    ('sadness', 'high'): 'Despair', ('sadness', 'mid'): 'Sadness', ('sadness', 'low'): 'Melancholy',
    ('joy', 'high'): 'Excitement', ('joy', 'mid'): 'Happiness', ('joy', 'low'): 'Contentment',
    ('fear', 'high'): 'Panic', ('fear', 'mid'): 'Fear', ('fear', 'low'): 'Anxiety',
    ('love', 'high'): 'Passion', ('love', 'mid'): 'Love', ('love', 'low'): 'Warmth',
    ('surprise', 'high'): 'Shock', ('surprise', 'mid'): 'Surprise', ('surprise', 'low'): 'Curiosity',
    ('angry', 'high'): 'Rage', ('angry', 'mid'): 'Anger', ('angry', 'low'): 'Irritation',
    ('sad', 'high'): 'Despair', ('sad', 'mid'): 'Sadness', ('sad', 'low'): 'Melancholy',
    ('happy', 'high'): 'Excitement', ('happy', 'mid'): 'Happiness', ('happy', 'low'): 'Contentment'
}

# ======= FOCAL LOSS DEFINITION =======
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
# =================== RECORD AUDIO + VIDEO ===================
def record_audio_video():
    recording = True

    def audio_recorder():
        audio_data = []
        def callback(indata, frames, time_info, status):
            if not recording:
                raise sd.CallbackStop()
            audio_data.append(indata.copy())
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', callback=callback):
            while recording:
                sd.sleep(100)
        audio_array = np.concatenate(audio_data, axis=0)
        sf.write(RAW_AUDIO_FILE, audio_array, SAMPLE_RATE)

        # Preprocessing: normalize + denoise
        y, sr = librosa.load(RAW_AUDIO_FILE, sr=SAMPLE_RATE)
        y = librosa.util.normalize(y) * 0.8
        y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.15)
        sf.write(PROC_AUDIO_FILE, y, sr)

    def video_recorder():
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(VIDEO_FILE, fourcc, FPS, FRAME_SIZE)

        while recording:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, FRAME_SIZE)
                out.write(frame)
                cv2.imshow('Recording - Press SPACE to stop', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    audio_thread = threading.Thread(target=audio_recorder)
    video_thread = threading.Thread(target=video_recorder)

    audio_thread.start()
    video_thread.start()

    while True:
        if keyboard.is_pressed('space'):
            recording = False
            break
        time.sleep(0.1)

    audio_thread.join()
    video_thread.join()

    # Merge audio and video using ffmpeg
    merge_command = f'ffmpeg -y -i {VIDEO_FILE} -i {PROC_AUDIO_FILE} -c:v copy -c:a aac {FINAL_OUTPUT_FILE}'
    subprocess.call(merge_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("✅ Recording complete and merged.")

# =================== EMOTION ANALYZER CLASS ===================
class EmotionAnalyzer:
    def __init__(self):
        self.audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH, custom_objects={'FocalLoss': FocalLoss})
        self.tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL_PATH)
        self.text_model = BertForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
        self.text_model.eval()
        self.whisper_model = WhisperModel("base")
        self.video_model = load_model(VIDEO_MODEL_PATH)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.speaker_history = []

    def transcribe_audio(self):
        segments, _ = self.whisper_model.transcribe(PROC_AUDIO_FILE)
        transcription = ' '.join(segment.text for segment in segments).strip()
        with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
            f.write(transcription)
        return transcription

    def analyze_text_emotion(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze()
            idx = torch.argmax(probs).item()
            return TEXT_EMOTIONS[idx], probs[idx].item() * 100, probs.numpy()

    def extract_audio_features(self, audio_path):
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        y = librosa.util.normalize(y) * 0.85
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

    def analyze_audio_emotion(self):
        features = self.extract_audio_features(PROC_AUDIO_FILE)
        probs = self.audio_model.predict(features, verbose=0)[0]
        index = np.argmax(probs)
        label = AUDIO_EMOTIONS.get(f"{index+1:02d}", 'unknown')
        confidence = probs[index] * 100
        return label, confidence, probs

    def analyze_video_emotion(self, num_frames=30):
        cap = cv2.VideoCapture(FINAL_OUTPUT_FILE)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        frame_nums = []
        raw_frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            raw_frames.append(frame.copy())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces):
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                face = gray[y:y+h, x:x+w]
            else:
                face = gray  # fallback if no face detected

            resized = cv2.resize(face, (48, 48))  # ✅ FIXED: resize face, not entire frame
            norm = resized.astype('float32') / 255.0
            frames.append(norm)
            frame_nums.append(int(idx))

        cap.release()

        if not frames:
            return "No faces detected", 0.0, None

        predictions = []
        for f, frame_idx in zip(frames, frame_nums):
            input_tensor = f[np.newaxis, ..., np.newaxis]  # (1, 48, 48, 1)
            prob = self.video_model.predict(input_tensor, verbose=0)[0]
            idx = np.argmax(prob)
            emotion = VIDEO_EMOTIONS[idx]
            confidence = prob[idx]
            predictions.append((frame_idx, emotion, confidence, idx))

        print("\n🎥 Frame-by-frame video emotion predictions:")
        for i, (frame_idx, emotion, conf, _) in enumerate(predictions):
            print(f"Frame {i+1:02d} [Index {frame_idx:04d}]: {emotion} ({conf:.2f})")

        emotions_only = [e for _, e, _, _ in predictions]
        final_emotion = max(set(emotions_only), key=emotions_only.count)
        matching_confs = [c for _, e, c, _ in predictions if e == final_emotion]
        final_conf = np.mean(matching_confs) * 100  # Convert to %

        return final_emotion, final_conf, predictions



    def calculate_confidence_entropy(self, probs):
        try:
            return 1.0 - entropy(probs) / np.log(len(probs))
        except:
            return 0.5

    def calculate_rms(self, audio_path):
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        rms = librosa.feature.rms(y=y)
        return np.mean(rms)

    def calculate_adaptive_intensity(self, rms_value):
        self.speaker_history.append(rms_value)
        if len(self.speaker_history) < 5:
            if rms_value < 0.015: return 'low'
            if rms_value < 0.08: return 'mid'
            return 'high'
        low_thresh = np.percentile(self.speaker_history, 33)
        mid_thresh = np.percentile(self.speaker_history, 66)
        if rms_value < low_thresh: return 'low'
        elif rms_value < mid_thresh: return 'mid'
        else: return 'high'

    def text_prioritized_fusion(self, text_emotion, audio_emotion, text_conf, audio_conf,
                                text_probs, audio_probs, intensity, transcription):
        text_certainty = self.calculate_confidence_entropy(text_probs)
        audio_certainty = self.calculate_confidence_entropy(audio_probs)

        is_threat = bool(threatening_patterns.search(transcription))
        is_pos = bool(positive_patterns.search(transcription))
        is_neg = bool(negative_patterns.search(transcription))

        pos_emotions = {'joy', 'love', 'happy', 'calm'}
        neg_emotions = {'sadness', 'anger', 'fear', 'sad', 'angry', 'fearful', 'disgust'}
        text_category = 'positive' if text_emotion in pos_emotions else 'negative' if text_emotion in neg_emotions else 'neutral'
        audio_category = 'positive' if audio_emotion in pos_emotions else 'negative' if audio_emotion in neg_emotions else 'neutral'

        if text_conf > 95.0:
            base_emotion = text_emotion
            confidence = text_conf
            reason = f"High text confidence ({text_conf:.1f}%)"
        elif text_conf > 85.0 and ((text_category == 'negative' and is_neg) or (text_category == 'positive' and is_pos)):
            base_emotion = text_emotion
            confidence = text_conf
            reason = "Text confidence + language pattern match"
            
        elif text_category != audio_category and text_category != 'neutral' and audio_category != 'neutral':
            confidence_diff = abs(text_conf - audio_conf)
            if text_conf > 80.0 and confidence_diff > 15:
                base_emotion = text_emotion
                confidence = text_conf
                reason = "Text dominance on disagreement"

            else:
                base_emotion = text_emotion
                confidence = text_conf * 0.7 + audio_conf * 0.3
                reason = "Text-weighted resolution"
        else:
            text_weight = 0.7 + (text_certainty * 0.2)
            audio_weight = 1 - text_weight
            if text_conf > 70.0:
                base_emotion = text_emotion
                confidence = text_conf * text_weight + audio_conf * audio_weight
                reason = "Text-weighted fusion"
            else:
                base_emotion = audio_emotion
                confidence = audio_conf
                reason = "Low text confidence, using audio"

        refined_emotion = emotion_mapping.get((base_emotion.lower(), intensity), base_emotion.title())
        return refined_emotion, confidence, False, reason
    
def run_pipeline():
    print("===== FULL MULTIMODAL PIPELINE READY =====")
    analyzer = EmotionAnalyzer()

    while True:
        cmd = input("\nType 'rec' to record or 'exit' to quit:\n> ").strip().lower()
        if cmd == 'exit':
            return None  # or break/exit

        elif cmd == 'rec':
            print("🎙️ Starting recording... Press SPACE to stop.")
            record_audio_video()

            print("🔍 Transcribing...")
            transcription = analyzer.transcribe_audio()
            print(f"📝 Transcription: {transcription}")

            print("🔎 Analyzing text emotion...")
            text_em, text_conf, text_probs = analyzer.analyze_text_emotion(transcription)

            print("🔊 Analyzing audio emotion...")
            audio_em, audio_conf, audio_probs = analyzer.analyze_audio_emotion()

            rms_value = analyzer.calculate_rms(PROC_AUDIO_FILE)
            intensity = analyzer.calculate_adaptive_intensity(rms_value)

            print("🎛️ Fusing audio and text...")
            fused_emotion, fused_conf, disagreement, reason = analyzer.text_prioritized_fusion(
                text_em, audio_em, text_conf, audio_conf, text_probs, audio_probs, intensity, transcription
            )

            print("🎥 Analyzing video emotion...")
            video_emotion, video_conf, video_probs = analyzer.analyze_video_emotion()

            print("🔗 Late fusion with confidence weighting...")

            # Count how often each emotion appeared in video
            video_counts = {}
            for _, emo, _, _ in video_probs:
                video_counts[emo] = video_counts.get(emo, 0) + 1

            most_common_video_emotion = max(video_counts, key=video_counts.get)
            dominance_ratio = video_counts[most_common_video_emotion] / len(video_probs)
            is_video_dominant = (most_common_video_emotion == video_emotion) and video_conf >= 50 and dominance_ratio >= 0.5
            print(video_counts[most_common_video_emotion])
            if is_video_dominant and video_emotion.lower() != 'neutral':
                final_emotion = video_emotion.lower()
                final_conf = video_conf
                source = "Video (dominant)"
                reason += " + Dominant video emotion over frames"
            else:
                # Fallback to Audio+Text if video is neutral
                if video_emotion.lower() == 'neutral':
                    final_emotion = fused_emotion
                    final_conf = fused_conf
                    source = "Audio+Text (fallback from neutral video)"
                    reason += " + Video predicted neutral; using Audio+Text instead"
                elif fused_conf >= video_conf:
                    final_emotion = fused_emotion
                    final_conf = fused_conf
                    source = "Audio+Text"
                else:
                    final_emotion = video_emotion.lower()
                    final_conf = video_conf
                    source = "Video"



            print(f"\n===== FINAL RESULT =====")
            print(f"Audio+Text: {fused_emotion} ({fused_conf:.1f}%)")
            print(f"Video: {video_emotion} ({video_conf:.1f}%)")
            print(f"✅ Final Fused Emotion: {final_emotion} ({final_conf:.1f}%) [Source: {source}]")
            print(f"🛠️ Reason: {reason}")
            if disagreement:
                print("⚠️ Disagreement detected between modalities.")
            print("=========================\n")

            return final_emotion  # ✅ <-- Return it here

if __name__ == '__main__':
    emotion = run_pipeline()
    print(f"\n🎯 Emotion returned: {emotion}")
