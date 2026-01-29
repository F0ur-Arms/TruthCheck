import whisper

model = whisper.load_model("small")

def extract_from_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]
