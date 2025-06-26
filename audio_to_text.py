import whisper
import os
import argparse

def transcribe_wav(wav_path, model_size="small"):
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"{wav_path} does not exist")

    # Load Whisper model
    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)

    # Transcribe audio
    print(f"Transcribing '{wav_path}'...")
    result = model.transcribe(wav_path)

    # Save transcription to .txt file
    base = os.path.splitext(wav_path)[0]
    txt_path = base + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"Transcription saved to: {txt_path}")
    return txt_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe a .wav file to .txt using Whisper")
    parser.add_argument("wav_file", help="Path to the input .wav file")
    parser.add_argument("--model", default="small", help="Whisper model size: tiny, base, small, medium, large")

    args = parser.parse_args()
    transcribe_wav(args.wav_file, args.model)
