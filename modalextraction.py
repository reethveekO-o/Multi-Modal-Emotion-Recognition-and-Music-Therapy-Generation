import cv2
import sounddevice as sd
import soundfile as sf
import threading
import numpy as np
import subprocess
import os
import keyboard
import time
from faster_whisper import WhisperModel
import librosa
import noisereduce as nr

# === CONFIG ===
session_folder = "RecordedSession"
os.makedirs(session_folder, exist_ok=True)

video_filename = os.path.join(session_folder, "output_video.avi")
audio_filename = os.path.join(session_folder, "output_audio.wav")
final_output = os.path.join(session_folder, "final_output.mp4")
final_transcript = os.path.join(session_folder, "final_output.txt")

fps = 20
frame_size = (640, 480)
samplerate = 44100
channels = 1

recording = True

def record_audio():
    recording_audio = []

    def callback(indata, frames, time_info, status):
        if not recording:
            raise sd.CallbackStop()
        recording_audio.append(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=channels, dtype='int16', callback=callback):
        while recording:
            sd.sleep(100)

    audio_array = np.concatenate(recording_audio, axis=0)
    sf.write(audio_filename, audio_array, samplerate)

    # === APPLY PREPROCESSING AFTER SAVING RAW ===
    y, sr = librosa.load(audio_filename, sr=samplerate)
    y = librosa.util.normalize(y) * 0.8  # Light normalization without crushing dynamics
    y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.15)
    sf.write(audio_filename, y, sr)
    print(f"✅ Preprocessed and saved: {audio_filename}")

def record_video():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

    while recording:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, frame_size)
            out.write(frame)
            cv2.imshow('Recording - Press SPACE to stop', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def merge_audio_video(video_file, audio_file, output_file):
    command = f'ffmpeg -y -loglevel error -i "{video_file}" -i "{audio_file}" -c:v copy -c:a aac -strict experimental "{output_file}"'
    subprocess.call(command, shell=True)

def transcribe_wav(wav_path, model_size="base"):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(wav_path)
    transcription = ' '.join(segment.text for segment in segments)
    with open(final_transcript, "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"✅ Transcription saved: {final_transcript}")

if __name__ == "__main__":
    audio_thread = threading.Thread(target=record_audio)
    video_thread = threading.Thread(target=record_video)

    recording = True
    audio_thread.start()
    video_thread.start()

    while True:
        if keyboard.is_pressed('space'):
            recording = False
            break
        time.sleep(0.1)

    audio_thread.join()
    video_thread.join()

    merge_audio_video(video_filename, audio_filename, final_output)
    transcribe_wav(audio_filename, model_size="base")

    print(f"\n🎉 Done: {final_output} + {final_transcript}")
