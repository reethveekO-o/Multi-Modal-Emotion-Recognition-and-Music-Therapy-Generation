from faster_whisper import WhisperModel
import os
import argparse

def transcribe_wav(wav_path, model_size="base"):
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"{wav_path} does not exist")

    # Load Faster-Whisper model
    print(f"🔄 Loading Faster-Whisper model ({model_size})...")
    model = WhisperModel(model_size)

    # Transcribe audio
    print(f"🎙️ Transcribing '{wav_path}'...")
    segments, _ = model.transcribe(wav_path)

    # Combine all segments into one transcript
    transcription = ' '.join(segment.text for segment in segments)

    # Save transcription to .txt file
    base = os.path.splitext(wav_path)[0]
    txt_path = base + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcription)

    print(f"✅ Transcription saved to: {txt_path}")
    return txt_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe a .wav file to .txt using Faster-Whisper")
    parser.add_argument("wav_file", help="Path to the input .wav file")
    parser.add_argument("--model", default="base", help="Model size: tiny, base, small, medium, large-v2")

    args = parser.parse_args()
    transcribe_wav(args.wav_file, args.model)
