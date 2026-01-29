from pipeline.asr import extract_from_audio
from pipeline.normalize import normalize_text
from pipeline.lang_detect import detect_language
from pipeline.hinglish import is_hinglish

audio_path = "data/raw_inputs/sample_audio.mp4"

raw_text = extract_from_audio(audio_path)
normalized_text = normalize_text(raw_text)

print("ASR RAW TEXT:\n", raw_text)
print("\nNORMALIZED TEXT:\n", normalized_text)


lang, conf = detect_language(normalized_text)
hinglish_flag = is_hinglish(normalized_text)

print("\nLANGUAGE:", lang, "CONFIDENCE:", round(conf, 2))
print("HINGLISH:", hinglish_flag)
