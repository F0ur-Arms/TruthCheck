import re

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[!?]{2,}', '!', text)
    return text.strip()
