# from googletrans import Translator
# import re

# class HinglishMapper:
#     def __init__(self):
#         self.translator = Translator()

#     def clean_text(self, text):
#         #added basic cleans
#         text = text.lower().strip()
#         #rem links
#         text = re.sub(r'http\S+|www\S+', '', text)
#         #remove emos
#         text = re.sub(r'[^\w\s,.!?]', '', text)
#         #remove rep chars
#         text = re.sub(r'(.)\1{2,}', r'\1\1', text)
#         text = text.strip()
        
#         # 2. Translate Hinglish to English
#         try:
#             # Google detects Hinglish as Hindi and converts to English
#             translation = self.translator.translate(text, dest='en')
#             cleaned = translation.text
#             print(f"--- [Translation] Input: {text} -> Output: {cleaned} ---")
#             return cleaned
#         except Exception as e:
#             print(f"--- [Translation Error] {e}. Using raw text. ---")
#             return text
        

#this is just for running since translate wasnt working 
import re

class HinglishMapper:
    def __init__(self):
        # translation disabled (unstable in googletrans newer versions)
        pass

    def clean_text(self, text):
        """
        Cleans noisy Hinglish text but avoids unreliable online translation.
        Keeps semantic meaning for embedding + NLI.
        """

        # lowercase
        text = text.lower().strip()

        # remove links
        text = re.sub(r'http\S+|www\S+', '', text)

        # remove emojis / symbols but keep punctuation
        text = re.sub(r'[^\w\s,.!?]', '', text)

        # remove excessive repeated characters
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text
