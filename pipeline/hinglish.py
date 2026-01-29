HINDI_ROMAN_WORDS = {
    "hai", "nahi", "nahin", "mat", "lo", "hota", "hoti", "karta",
    "karte", "kyu", "kyon", "isse", "usse", "acha", "bura", "bilkul"
}

def is_hinglish(text):
    words = text.lower().split()
    hindi_hits = sum(1 for w in words if w in HINDI_ROMAN_WORDS)
    english_hits = sum(1 for w in words if w.isascii() and w.isalpha())
    return hindi_hits >= 2 and english_hits >= 2
