Fine. Here’s a **clean, professional README** you can drop straight into GitHub.
No fluff, no resume nonsense, no future promises. This documents the project **up to Step 4 only**, exactly as requested.

You can copy-paste this into `README.md`.

---

# TruthCheck – Multimodal Health Misinformation Pipeline (Phase 1)

This repository contains the **multimodal preprocessing pipeline** for an Indian health misinformation detection system.
It supports **text, image, and voice inputs** and normalizes them into structured text with **language detection and Hinglish tagging**.

⚠️ This repository covers the project **up to Step 4 (Preprocessing + Language Tagging)**.
No classification or ML models are included yet.

---

## 📌 Features Implemented (Up to Step 4)

* Text input handling
* OCR for images (English + Hindi)
* ASR for voice notes (Hindi / Hinglish / English)
* Text normalization
* Language detection (fastText)
* Hinglish tagging (rule-based)

---

## 🧱 Project Structure

```
TruthCheck/
│
├── data/
│   ├── raw_inputs/        # Input images / audio files
│   └── extracted_text/   # (Optional) saved outputs
│
├── logs/                 # Debug and pipeline logs
│
├── models/
│   └── language/
│       └── lid.176.bin   # fastText language ID model
│
├── pipeline/
│   ├── text_input.py     # Text passthrough
│   ├── ocr.py            # Image → Text (Tesseract)
│   ├── asr.py            # Audio → Text (Whisper)
│   ├── normalize.py      # Text normalization
│   ├── lang_detect.py    # Language detection
│   └── hinglish.py       # Hinglish tagging
│
├── main.py               # Entry point for testing pipeline
└── README.md
```

---

## 🛠 Requirements

### System Requirements

* **Windows 10/11**
* **Python 3.11.x** (recommended)
* **FFmpeg** (required for ASR)

---

### Python Packages

Install all dependencies inside a virtual environment:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper
pip install pytesseract pillow
pip install fasttext
pip install scikit-learn
pip install "numpy<2"
```

---

### External Dependencies

#### 1️⃣ Tesseract OCR

* Download from: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
* Install **English** and **Hindi** language packs
* Note installation path (used in `ocr.py`)

Default path used:

```
C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

#### 2️⃣ FFmpeg

* Download from: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
* Add FFmpeg to system PATH
* Verify installation:

```bash
ffmpeg -version
```

---

#### 3️⃣ fastText Language Model

Download the pretrained language identification model:

👉 [https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin)

Place it at:

```
models/language/lid.176.bin
```

---

## 🚀 How to Run (Up to Step 4)

1️⃣ Create and activate virtual environment:

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

2️⃣ Select the venv interpreter in VS Code:

```
.\venv\Scripts\python.exe
```

3️⃣ Place sample inputs:

* Image → `data/raw_inputs/sample_image.png`
* Audio → `data/raw_inputs/sample_audio.mp4`

4️⃣ Run the pipeline:

```bash
python main.py
```

---

## ✅ Expected Output

The pipeline prints:

* Raw ASR / OCR text
* Normalized text
* Detected language
* Hinglish flag

Example:

```
ASR RAW TEXT:
 प्रोटीन मतलो ...

NORMALIZED TEXT:
 प्रोटीन मतलो ...

LANGUAGE: hi
CONFIDENCE: 1.0
HINGLISH: False
```

---

## ⚠️ Known Warnings (Safe to Ignore)

```
FP16 is not supported on CPU; using FP32 instead
```

This is expected when running Whisper on CPU and does not affect correctness.

---

## 📍 Current Project Status

✔ Multimodal preprocessing complete
✔ Language detection & Hinglish tagging complete
⏳ Dataset ingestion and ML classification not included in this repo

---

## 🧠 Design Philosophy

* Real-world noisy inputs are preserved
* No aggressive grammar or spelling correction
* Language tagging instead of forced translation
* Clean separation between preprocessing and ML

---

## 📌 Next Steps (Not in this Repo)

* Dataset ingestion
* Baseline classifier (TF-IDF + Logistic Regression)
* Transformer fine-tuning (MuRIL / IndicBERT)
* Android Share Sheet integration

---

## 👤 Author

Built as part of an exploratory ML project focused on **health misinformation in the Indian context**.

---

If you want, next I can:

* write a **clean commit history plan**
* help you split this into **logical Git commits**
* draft a **professor-facing explanation doc**
* or move to **Step 6 (actual ML)**
