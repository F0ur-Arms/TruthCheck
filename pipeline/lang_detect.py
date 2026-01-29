import fasttext

model = fasttext.load_model("models/language/lid.176.bin")

def detect_language(text):
    text = text.replace("\n", " ")
    labels, scores = model.predict(text)
    lang = labels[0].replace("__label__", "")
    confidence = scores[0]
    return lang, confidence
