import os
import pytesseract
from PIL import Image

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def extract_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(
        img,
        lang="eng+hin"
    )
    return text
