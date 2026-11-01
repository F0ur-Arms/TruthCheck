"""KB batch 2: Indian guidelines from PDFs (ICMR, NIN) and Hindi Vishvas News health articles.

PDFs are kept out of the repo: only prose blocks are extracted to .txt, so table and flowchart
fragments never become evidence. PDFs that disallow copying are skipped.

usage: python scripts/kb/scrape_batch2.py PDF_CACHE_DIR [--sample]
needs: beautifulsoup4, pymupdf
"""
import csv
import datetime
import glob
import os
import random
import re
import sys
import time
from urllib.parse import urljoin, urlparse

import pandas as pd
import pymupdf
import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
KB = "data/medical_kb"
CACHE = sys.argv[1]
SAMPLE = "--sample" in sys.argv
os.makedirs(CACHE, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (compatible; TruthCheckResearchBot/0.1; academic health misinformation project)"}
TODAY = datetime.date.today().isoformat()
RULE = "─" * 60
ICMR_FILES = "https://www.icmr.gov.in/icmrobject/custom_data/pdf/resource-guidelines/"
DOCS = [
    ("icmr_type-2-diabetes-guidelines-2018", "ICMR Guidelines for Management of Type 2 Diabetes 2018", "ICMR_GuidelinesType2diabetes2018_0.pdf"),
    ("icmr_type-1-diabetes-guidelines", "ICMR Guidelines for Management of Type 1 Diabetes", "ICMR_Guidelines_for_Management_of_Type_1_Diabetes.pdf"),
    ("icmr_index-tb-guidelines", "INDEX-TB Guidelines on extra-pulmonary tuberculosis for India", "Index-TB Guidelines - green colour 2594164.pdf"),
    ("icmr_probiotics-guidelines", "ICMR-DBT Guidelines for Evaluation of Probiotics in Food", "PROBIOTICS_GUIDELINES.pdf"),
    ("icmr_rickettsial-diseases-guidelines", "DHR-ICMR Guidelines on Rickettsial Diseases", "DHR-ICMR Guidelines on Ricketesial Diseases.pdf"),
]
NIN_BOOKLET = "https://www.nin.res.in/downloads/DGI_Booklet_English_CMYK.pdf"
STOP_HEAD = re.compile(r"^(references?|bibliography|annexure|appendix|list of (contributors|participants)|acknowledg|contributors|members of|writing committee)\b", re.I)
MAX_WORDS = 1100


def norm(u):
    p = urlparse(u.strip())
    return (p.netloc.lower().removeprefix("www.") + p.path.rstrip("/").removesuffix("/index.html")).lower()


g = pd.read_csv("evaluation/golden_set/golden_v1.csv", low_memory=False)
BLOCK = {norm(u) for u in g.evidence_url.dropna().astype(str) if u.startswith("http")}
EXIST = set()
for f in glob.glob(os.path.join(KB, "*.txt")):
    for l in open(f, encoding="utf-8", errors="ignore").read(600).splitlines():
        if l.startswith("URL:"):
            EXIST.add(norm(l[4:]))


def get(u):
    time.sleep(random.uniform(1.5, 3.0))
    return requests.get(u, headers=H, timeout=120)


def fetch_pdf(u):
    path = os.path.join(CACHE, u.rsplit("/", 1)[1])
    if not os.path.exists(path):
        r = get(u.replace(" ", "%20"))
        if r.status_code != 200 or r.content[:4] != b"%PDF":
            print("PDF failed", r.status_code, u)
            return None
        open(path, "wb").write(r.content)
    d = pymupdf.open(path)
    if not d.permissions & pymupdf.PDF_PERM_COPY:
        print("PDF disallows copying, skipped", u)
        return None
    return d


def write(fn, label, title, url, pub, body):
    txt = f"SOURCE: {label}\nTITLE: {title}\nURL: {url}\n{RULE}\n\n{body}\n"
    if SAMPLE:
        print("=" * 20, fn, len(body.split()))
        print(txt[:800])
        print("... END:", body[-200:])
        return
    open(os.path.join(KB, fn), "w", encoding="utf-8", newline="\n").write(txt)
    csvp = os.path.join(KB, "SOURCES.csv")
    with open(csvp, "a", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow([fn, pub, url, TODAY])
    print("ok", len(body.split()), fn)


BULLET = re.compile(r"^[\s•▪◦\-–*]+|^z$")
HEAD = re.compile(r"^(\d+(\.\d+)*\.?\s+[A-Z]|chapter\s+\d|section\s+\d|[A-Z][A-Z ,&/-]{6,}$)")


def paragraphs(d):
    """Rebuild (kind, text) paragraphs from PDF lines, without running headers and table fragments.

    PDF lines break mid-sentence, so lines are joined until a sentence ends. A paragraph must read as
    prose: 12+ words, ending in punctuation, with full-looking lines. Table cells fail that test.
    """
    pages = [[l.strip() for l in p.get_text("text").split("\n")] for p in d]
    seen = {}
    for ls in pages:
        for x in set(ls):
            key = re.sub(r"\d+", "", x)
            seen[key] = seen.get(key, 0) + 1
    repeated = {k for k, n in seen.items() if n >= 4 and k}
    out, buf = [], []

    def flush():
        if buf:
            text = " ".join(buf)
            words = len(text.split())
            if words >= 12 and words / len(buf) >= 5 and text[-1] in ".?!)" and not re.search(r"\.{5,}", text):
                out.append(("para", re.sub(r"\s+", " ", text)))
            buf.clear()

    for ls in pages:
        for i, x in enumerate(ls):
            if not x or re.fullmatch(r"[\divxlc]+", x, re.I) or re.sub(r"\d+", "", x) in repeated or x == "z":
                continue
            if HEAD.match(x) and len(x.split()) <= 12:
                flush()
                out.append(("head", x))
                continue
            if BULLET.match(x):
                flush()
                x = BULLET.sub("", x).strip()
                if not x:
                    continue
            buf.append(x)
            nxt = next((n for n in ls[i + 1:] if n), "")
            if x[-1] in ".?!" and (not nxt or nxt[0].isupper() or nxt[0].isdigit() or BULLET.match(nxt)):
                flush()
    flush()
    return out


def split_doc(prefix, title, url):
    d = fetch_pdf(url)
    if d is None:
        return
    parts, cur, head = [], [], None
    for kind, x in paragraphs(d):
        if kind == "head":
            if STOP_HEAD.match(re.sub(r"^[\d.\s]+", "", x)) and sum(len(c.split()) for p in parts for c in p[1]) > 500:
                break
            if sum(len(c.split()) for c in cur) > MAX_WORDS * 0.5:
                parts.append((head, cur))
                cur, head = [], x
            elif not cur:
                head = x
            cur.append(x)
            continue
        if re.match(r"^(Table|Figure|Fig\.)\s*\d", x):
            continue
        cur.append(x)
        if sum(len(c.split()) for c in cur) > MAX_WORDS:
            parts.append((head, cur))
            cur = []
    if cur:
        parts.append((head, cur))
    n = 0
    for h, blocks in parts:
        body = "\n".join(blocks)
        if len(body.split()) < 150:
            continue
        n += 1
        t = f"{title}: {h.title() if h.isupper() else h}" if h else f"{title}, part {n}"
        write(f"{prefix}-{n:02d}.txt", "ICMR — Guidelines", t, url, "Indian Council of Medical Research", body)
    print(prefix, "files", n, "from pages", len(d))


def nin_booklet():
    d = fetch_pdf(NIN_BOOKLET)
    if d is None:
        return
    # The booklet's spreads mix guidelines on a page, so it stays one file rather than per-guideline files.
    lines = []
    for p in d:
        for b in p.get_text("blocks"):
            x = re.sub(r"\s+", " ", b[4]).strip()
            if b[6] == 0 and len(x.split()) >= 5 and not re.match(r"^(Points to register|Dietary Guidelines|Let.s (know our|commit)|Adopt the DIETARY|SCAN TO|for Indians Nutrition Atlas|My Plate for the Day|Indian Food Composition|e-modules)", x, re.I):
                lines.append(x)
    write("nin_dietary-guidelines-2024-booklet.txt", "ICMR-NIN — Dietary Guidelines for Indians",
          "Dietary Guidelines for Indians 2024: summary booklet", NIN_BOOKLET, "ICMR-National Institute of Nutrition",
          "\n".join(dict.fromkeys(lines)))


CUT = re.compile(r"^(claim review|claimed by|fact check\s*:|also read|यह भी पढ़ें|ये भी पढ़ें|tags?|टैग)\b", re.I)
STOP_HI = re.compile(r"(पूरा सच जानें|Related Fact Check|फर्जी खबरों के खिलाफ|सच की इस लड़ाई|समर्थन करें|सहयोग करें|सहयोग दें|हमारी मदद करें|Help us in our fight)")
DROP = ("nav,header,footer,aside,script,style,form,button,noscript,svg,iframe,figure,.share,.social,.breadcrumb,"
        "[role=navigation],.related-posts,.sharedaddy")


def hindi_share(t):
    letters = [c for c in t if c.isalpha()]
    return sum(0x900 <= ord(c) <= 0x97f for c in letters) / max(1, len(letters))


def extract(html):
    sp = BeautifulSoup(html, "html.parser")
    h1 = sp.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""
    root = None
    for s in (".view-full", ".vn-content", "article", ".entry-content", "main"):
        root = sp.select_one(s)
        if root and len(root.get_text().split()) > 60:
            break
    root = root or sp.body
    for bad in root.select(DROP):
        bad.decompose()
    out = []
    for e in root.find_all(["h2", "h3", "h4", "p", "li"]):
        if e.name == "li" and e.find(["p", "li", "ul", "ol"]):
            continue
        x = re.sub(r"\s+", " ", e.get_text(" ", strip=True)).strip()
        if STOP_HI.search(x):
            break
        if not x or x == title or CUT.match(x):
            continue
        out.append(("- " + x) if e.name == "li" else x)
    return title, "\n".join(dict.fromkeys(out))


def vishvas_hindi(cap=70):
    urls = []
    for n in range(1, 13):
        u = "https://www.vishvasnews.com/health/" + (f"page/{n}/" if n > 1 else "")
        r = get(u)
        if r.status_code != 200:
            print("listing", r.status_code, u)
            break
        sp = BeautifulSoup(r.text, "html.parser")
        found = [urljoin(u, a["href"]).split("#")[0] for a in sp.find_all("a", href=True)]
        found = [x for x in found if re.match(r"^https://www\.vishvasnews\.com/health/[^/]+/$", x) and "/page/" not in x]
        urls = list(dict.fromkeys(urls + found))
        print("listing page", n, "total", len(urls))
    kept = 0
    for u in urls:
        if kept >= (3 if SAMPLE else cap):
            break
        if norm(u) in BLOCK or norm(u) in EXIST:
            continue
        r = get(u)
        if r.status_code != 200:
            print(r.status_code, u)
            continue
        r.encoding = "utf-8"
        title, body = extract(r.text)
        w, share = len(body.split()), hindi_share(body)
        if w < 150 or share < 0.6:
            print("skip", w, round(share, 2), u)
            continue
        slug = re.sub(r"[^a-z0-9-]+", "-", urlparse(u).path.lower().split("/")[-2])[:75].strip("-") or f"article-{kept}"
        write(f"vishvas__hi-{slug}.txt", "Vishvas News — स्वास्थ्य फैक्ट चेक", title, u, "Vishvas News", body)
        kept += 1
    print("hindi kept", kept)


nin_booklet()
for prefix, title, name in DOCS[:1] if SAMPLE else DOCS:
    split_doc(prefix, title, ICMR_FILES + name)
vishvas_hindi()
print("DONE")
