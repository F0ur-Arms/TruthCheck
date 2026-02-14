from googletrans import Translator
import re

class HinglishMapper:
    def __init__(self):
        self.translator = Translator()

    def clean_text(self, text):
        # 1. Basic cleanup
        text = text.strip()
        
        # 2. Translate Hinglish to English
        try:
            # Google detects Hinglish as Hindi and converts to English
            translation = self.translator.translate(text, dest='en')
            cleaned = translation.text
            print(f"--- [Translation] Input: {text} -> Output: {cleaned} ---")
            return cleaned
        except Exception as e:
            print(f"--- [Translation Error] {e}. Using raw text. ---")
            return text