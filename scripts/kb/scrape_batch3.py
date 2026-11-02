"""KB batch 3: Indian government sources that allow reuse with acknowledgement.

NHM immunisation documents, Central TB Division FAQs, NCVBDC disease pages, NCDC advisories and
FSSAI clarifications. Only prose is kept, so tables and flowcharts never become evidence. Third-party
documents (the WHO leaflet on FSSAI's page) and PDFs that disallow copying are skipped.

usage: python scripts/kb/scrape_batch3.py PDF_CACHE_DIR b3_urls.json nhm_imm.json
needs: beautifulsoup4, pymupdf
"""
import csv
import datetime
import json
import os
import random
import re
import sys
import time
import zipfile
from urllib.parse import unquote

import pymupdf
import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
KB = "data/medical_kb"
CACHE, URLS, NHM = sys.argv[1:4]
os.makedirs(CACHE, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (compatible; TruthCheckResearchBot/0.1; academic health misinformation project)"}
TODAY = datetime.date.today().isoformat()
RULE = "─" * 60
MAX_WORDS = 1100
BULLET = re.compile(r"^[\s•▪◦\-–*]+")
HEAD = re.compile(r"^(\d+(\.\d+)*\.?\s+[A-Z]|chapter\s+\d|section\s+\d|[A-Z][A-Z ,&/-]{6,}$)")
STOP_HEAD = re.compile(r"^(references?|bibliography|annexure|appendix|acknowledg|contributors)\b", re.I)
urls = json.load(open(URLS))
done = {r["filename"] for r in csv.DictReader(open(os.path.join(KB, "SOURCES.csv"), encoding="utf-8"))}


def get(u):
    time.sleep(random.uniform(1.5, 3.0))
    return requests.get(u.replace(" ", "%20"), headers=H, timeout=120)


def write(fn, label, title, url, pub, body):
    if fn in done:
        return
    done.add(fn)
    open(os.path.join(KB, fn), "w", encoding="utf-8", newline="\n").write(f"SOURCE: {label}\nTITLE: {title}\nURL: {url}\n{RULE}\n\n{body}\n")
    with open(os.path.join(KB, "SOURCES.csv"), "a", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow([fn, pub, url, TODAY])
    print("ok", len(body.split()), fn)


def fetch_pdf(u):
    path = os.path.join(CACHE, re.sub(r"[^\w.-]+", "_", unquote(u.rsplit("/", 1)[1])))
    if not os.path.exists(path):
        try:
            r = get(u)
        except Exception as e:
            print("ERR", u, str(e)[:80])
            return None
        if r.status_code != 200 or r.content[:4] != b"%PDF":
            print("PDF failed", r.status_code, u)
            return None
        open(path, "wb").write(r.content)
    d = pymupdf.open(path)
    if not d.permissions & pymupdf.PDF_PERM_COPY:
        print("PDF disallows copying, skipped", u)
        return None
    return d


def paragraphs(d):
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
            if words >= 8 and words / len(buf) >= 4 and text[-1] in ".?!)" and not re.search(r"\.{5,}", text):
                out.append(("para", re.sub(r"\s+", " ", text)))
            buf.clear()

    for ls in pages:
        for i, x in enumerate(ls):
            if not x or re.fullmatch(r"[\divxlc]+", x, re.I) or re.sub(r"\d+", "", x) in repeated:
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


def split_write(prefix, title, url, items, label, pub):
    parts, cur = [], []
    for kind, x in items:
        if kind == "head" and STOP_HEAD.match(re.sub(r"^[\d.\s]+", "", x)) and parts:
            break
        if re.match(r"^(Table|Figure|Fig\.)\s*\d", x):
            continue
        cur.append(x)
        if sum(len(c.split()) for c in cur) > MAX_WORDS:
            parts.append(cur)
            cur = []
    if cur:
        parts.append(cur)
    parts = [p for p in parts if len(" ".join(p).split()) >= 150]
    for i, p in enumerate(parts, 1):
        t = title if len(parts) == 1 else f"{title}, part {i}"
        fn = f"{prefix}.txt" if len(parts) == 1 else f"{prefix}-{i:02d}.txt"
        write(fn, label, t, url, pub, "\n".join(p))
    print(prefix, "files", len(parts))


def do_pdf(prefix, title, url, label, pub):
    d = fetch_pdf(url)
    if d is not None:
        split_write(prefix, title, url, paragraphs(d), label, pub)


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:70]


# NHM immunisation
NHM_LABEL, NHM_PUB = "National Health Mission, MoHFW — Immunization", "National Health Mission, Ministry of Health and Family Welfare"
want = re.compile(r"FAQ_on_Immunization_for_(Parents|Health_Workers|other)|Immunization_Handbook_for_Health_Workers-English|HPV|PCV_Operational|Rotasiil|MCP_Guide|Paracetamol", re.I)
for u in json.load(open(NHM)):
    if want.search(u):
        name = re.sub(r"\.pdf$", "", unquote(u.rsplit("/", 1)[1]), flags=re.I)
        do_pdf("nhm_" + slug(name), name.replace("_", " ").replace("-", " "), u, NHM_LABEL, NHM_PUB)

# Central TB Division FAQs (docx, read as XML)
TB_LABEL, TB_PUB = "Central TB Division, MoHFW — FAQs", "Central TB Division, Ministry of Health and Family Welfare"
tbu = "https://tbcindia.mohfw.gov.in/wp-content/uploads/2023/04/Tuberculosis_FAQs_12_September_2022.docx"
r = get(tbu)
if r.status_code == 200:
    p = os.path.join(CACHE, "tb_faqs.docx")
    open(p, "wb").write(r.content)
    xml = zipfile.ZipFile(p).read("word/document.xml").decode("utf-8")
    paras = []
    for block in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        x = re.sub(r"\s+", " ", "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", block))).strip()
        if len(x.split()) >= 3:
            paras.append(("para", x.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")))
    split_write("mohfw_tb-faqs-2022", "Tuberculosis FAQs (September 2022)", tbu, paras, TB_LABEL, TB_PUB)
else:
    print("TB FAQ failed", r.status_code)

# NCDC advisories and public guidelines
NCDC_LABEL, NCDC_PUB = "National Centre for Disease Control, MoHFW", "National Centre for Disease Control, Ministry of Health and Family Welfare"
pick = re.compile(r"Guidelines on use of masks|Guidelines for Providing Home Care|Epidemiology of seasonal influenza|Vaccine Guidelines as on July 2022|Public health advisory Extreme heat|Health Advisory on Air Pollution|National Guidelines for Rabies Prophylaxis|National Guidelines on Diagnosis Case Management and Prevention and Control of Leptospiros|Public Health Guidelines for Floods", re.I)
for u, t in urls["ncdc"].items():
    if u.lower().endswith(".pdf") and pick.search(t):
        do_pdf("mohfw_ncdc-" + slug(t), re.sub(r"\s*\[.*$", "", t), u, NCDC_LABEL, NCDC_PUB)

# NCVBDC disease pages: short pages are merged per disease
VBD_LABEL, VBD_PUB = "National Centre for Vector Borne Diseases Control, MoHFW", "National Centre for Vector Borne Diseases Control, Ministry of Health and Family Welfare"
SKIP = re.compile(r"guideline|report|format|situation|data|tender|code", re.I)
DIS = re.compile(r"(?i)malaria|dengue|chikungunya|kala|filaria|encephalitis")
buckets = {}
for u, t in urls["vbd"].items():
    if "index4.php" not in u or SKIP.search(t):
        continue
    m = DIS.search(t)
    if not m:
        continue
    r = get(u)
    if r.status_code != 200:
        print(r.status_code, u)
        continue
    sp = BeautifulSoup(r.content, "html.parser")
    for bad in sp.select("nav,header,footer,script,style,form,aside"):
        bad.decompose()
    lines = [re.sub(r"\s+", " ", e.get_text(" ", strip=True)) for e in sp.find_all(["p", "li"]) if not e.find(["p", "li", "ul"])]
    lines = list(dict.fromkeys(x for x in lines if len(x.split()) >= 8 and not re.search(r"(?i)copyright|last updated|website|visitor|all rights", x)))
    if lines:
        buckets.setdefault(m.group().lower(), []).append((t, u, lines))
for dis, pages in buckets.items():
    body = "\n".join(f"{t}\n" + "\n".join(ls) for t, _, ls in pages)
    if len(body.split()) >= 150:
        write(f"mohfw_vbd-{slug(dis)}-overview.txt", VBD_LABEL, f"{dis.title()}: NCVBDC disease information", pages[0][1], VBD_PUB, body)
    else:
        print("short", dis, len(body.split()))

# FSSAI PDFs
FS_LABEL, FS_PUB = "FSSAI — Myth Buster", "Food Safety and Standards Authority of India"
BASE = "https://www.fssai.gov.in/docs/myth-buster/"
for name, title in [("FAQs_Get_Food_Fact_Right_Milk_06_05_2019.pdf", "FSSAI FAQ on milk and milk products"),
                    ("Clarification_Adulterated_Salt_24_04_2020.pdf", "FSSAI clarification on salt adulterated with detergent"),
                    ("Mythbuster_Salt_13_01_2020.pdf", "FSSAI information on ferrocyanides in salt")]:
    do_pdf("fssai_" + slug(name[:-4]), title, BASE + name, FS_LABEL, FS_PUB)

# Anaemia Mukt Bharat: report what is reachable, nothing is assumed
for u in ["https://www.anemiamuktbharat.info/resources/", "https://anemiamuktbharat.info/home/resources/"]:
    try:
        r = get(u)
        print("AMB", u, r.status_code, len(r.content), [a["href"] for a in BeautifulSoup(r.content, "html.parser").find_all("a", href=True) if ".pdf" in a["href"].lower()][:15])
    except Exception as e:
        print("AMB ERR", u, str(e)[:80])
print("DONE")
