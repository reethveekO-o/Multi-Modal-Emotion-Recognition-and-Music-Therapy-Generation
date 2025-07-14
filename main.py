import os
import logging
import warnings
import sys
# Suppress TensorFlow, xFormers, and general warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("xformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
import time
import threading
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from stress_final import analyze_video_stress as stress_check
from emotion_final import run_pipeline as emotion_check
# Spinner thread function
spinner_running = True
def spinner():
    while spinner_running:
        for ch in "|/-\\":
            print(f"\rGenerating... {ch}", end='', flush=True)
            time.sleep(0.1)

# Prompt builder
def generate_music_prompt(stress_score, emotion):
    if stress_score >= 0.66:
        stress_level = 'high'
        freq = '432'
    elif stress_score >= 0.33:
        stress_level = 'mid'
        freq = '528'
    else:
        stress_level = 'low'
        freq = '640'

    emotion_categories = {
        'rage': {'type': 'high_negative', 'synonyms': ['anger', 'irritation']},
        'anger': {'type': 'high_negative', 'synonyms': ['rage', 'annoyance', 'irritation']},
        'annoyance': {'type': 'mid_negative', 'synonyms': ['irritation', 'anger']},
        'irritation': {'type': 'mid_negative', 'synonyms': ['annoyance', 'anger']},
        'despair': {'type': 'low_negative', 'synonyms': ['sadness', 'melancholy']},
        'sadness': {'type': 'low_negative', 'synonyms': ['despair', 'melancholy']},
        'melancholy': {'type': 'low_negative', 'synonyms': ['sadness', 'despair']},
        'excitement': {'type': 'high_positive', 'synonyms': ['happiness']},
        'happiness': {'type': 'high_positive', 'synonyms': ['excitement', 'contentment']},
        'contentment': {'type': 'low_positive', 'synonyms': ['happiness']},
        'panic': {'type': 'high_fear', 'synonyms': ['fear', 'anxiety']},
        'fear': {'type': 'mid_fear', 'synonyms': ['panic', 'anxiety']},
        'anxiety': {'type': 'mid_fear', 'synonyms': ['fear', 'panic']},
        'passion': {'type': 'high_love', 'synonyms': ['love', 'warmth']},
        'love': {'type': 'mid_love', 'synonyms': ['passion', 'warmth']},
        'warmth': {'type': 'low_love', 'synonyms': ['love', 'passion']},
        'shock': {'type': 'high_surprise', 'synonyms': ['surprise']},
        'surprise': {'type': 'mid_surprise', 'synonyms': ['shock', 'curiosity']},
        'curiosity': {'type': 'low_surprise', 'synonyms': ['surprise']}
    }

    base_emotion = next((emo for emo in emotion_categories if emotion.lower() == emo or emotion.lower() in emotion_categories[emo]['synonyms']), 'contentment')
    emotion_type = emotion_categories[base_emotion]['type']

    therapy_static = {
        'high_negative':   'heavy and grounded with slow motion',
        'mid_negative':    'gentle and flowing',
        'low_negative':    'soft and emotionally warm',
        'high_positive':   'bright and rhythmic',
        'low_positive':    'peaceful and steady',
        'high_fear':       'deep and steady',
        'mid_fear':        'calm and predictable',
        'high_love':       'rich and emotional',
        'mid_love':        'soft and melodic',
        'low_love':        'light and soothing',
        'high_surprise':   'shifting but cohesive',
        'mid_surprise':    'quirky and interesting',
        'low_surprise':    'light and inquisitive'
    }

    tempo_ranges = {
        'high_negative': (40, 60 + (20 * (1 - stress_score))),
        'mid_negative':  (50, 70 + (10 * (1 - stress_score))),
        'low_negative':  (60, 80 + (10 * stress_score)),
        'high_positive': (100, 140 + (20 * stress_score)),
        'mid_positive':  (90, 120),
        'low_positive':  (70, 90),
        'high_fear':     (50, 70),
        'mid_fear':      (60, 80),
        'low_fear':      (60, 80),
        'high_love':     (70, 90),
        'mid_love':      (60, 80),
        'low_love':      (50, 70),
        'high_surprise': (80, 120),
        'mid_surprise':  (90, 130),
        'low_surprise':  (70, 100)
    }

    tempo_min, tempo_max = tempo_ranges[emotion_type]
    tempo = int(tempo_min + (tempo_max - tempo_min) * stress_score)
    duration = float(2 * 240 / tempo)
    print(f"Calculated music loop duration: {duration:} seconds")

    character = therapy_static[emotion_type]

    prompt= (
        f"A perfect loop of {character} instrumental, {tempo} BPM, centered around {freq} Hz drone, "
        "major mode with soft pads, with minimal percussion, long reverb, "
        "composed to reduce stress."
        "it fades in and fades out"
        " It starts and ends the same way."
    )
    return prompt, duration

# Get user input
video_path = r"C:\Users\rithvik\OneDrive\Documents\GitHub\Multi-Modal-Emotion-Recognition-and-Music-Therapy-Generation\RecordedSession\output_video.avi" 
emotion = emotion_check()
stressscore = stress_check(video_path, final_emotion=emotion)

# Generate music prompt
prompt, dur = generate_music_prompt(stressscore, emotion)
print(f"\nPrompt:\n{prompt}\n")
# Load model and generate music
print("Loading MusicGen model...")
model = MusicGen.get_pretrained(r"C:\Users\rithvik\OneDrive\Documents\GitHub\Multi-Modal-Emotion-Recognition-and-Music-Therapy-Generation\musicgen")
model.set_generation_params(duration=dur)
print("Model loaded.")

# Start spinner thread
spinner_thread = threading.Thread(target=spinner)
spinner_thread.start()
start_time = time.time()

descriptions = [prompt]
wav = model.generate(descriptions)

# Stop spinner
spinner_running = False
spinner_thread.join()

end_time = time.time()
print(f"\n\n✅ Music generated in {end_time - start_time:.2f} seconds.")

# Create output folder
output_folder = "generated_audio"
os.makedirs(output_folder, exist_ok=True)

# Save audio
for idx, one_wav in enumerate(wav):
    output_path = os.path.join(output_folder, f'{idx}.wav')
    audio_write(output_path, one_wav.cpu(), model.sample_rate, strategy="loudness")
print(f"🎵 Music loop saved as {output_folder}/0.wav")

# Target loop length (in seconds)
target_duration = dur * 4  # change to what you want
sample_rate = model.sample_rate

for idx, one_wav in enumerate(wav):
    # Original duration
    original_duration = one_wav.shape[-1] / sample_rate
    loops_needed = int(target_duration / original_duration)

    # Repeat waveform
    looped_wav = one_wav.repeat(1, loops_needed)

    # Save looped version
    looped_output_path = os.path.join(output_folder, f'{idx}_looped.wav')
    audio_write(looped_output_path, looped_wav.cpu(), sample_rate, strategy="loudness")
    print(f"🔁 Looped and saved as {looped_output_path} ({target_duration}s)")
