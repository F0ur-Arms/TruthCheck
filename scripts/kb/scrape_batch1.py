"""KB batch 1: select pages, skip leaks and existing URLs, write txt + SOURCES.csv.
usage: scrape.py urls_raw.json KB_DIR [--sample]"""
import sys, re, json, csv, time, random, glob, os, datetime
from collections import Counter
from urllib.parse import urlparse
import requests
import pandas as pd
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
raw = json.load(open(sys.argv[1]))
KB = sys.argv[2]
SAMPLE = "--sample" in sys.argv
H = {"User-Agent": "Mozilla/5.0 (compatible; TruthCheckResearchBot/0.1; academic health misinformation project)"}
TODAY = datetime.date.today().isoformat()


def norm(u):
    p = urlparse(u.strip())
    return (p.netloc.lower().removeprefix("www.") + p.path.rstrip("/").removesuffix("/index.html")).lower()


g = pd.read_csv("evaluation/golden_set/golden_v1.csv", low_memory=False)
BLOCK = {norm(u) for u in g.evidence_url.dropna().astype(str) if u.startswith("http")}
EXIST = set()
for f in glob.glob(os.path.join(KB, "*.txt")):
    try:
        for l in open(f, encoding="utf-8", errors="ignore").read(600).splitlines():
            if l.startswith("URL:"):
                EXIST.add(norm(l[4:]))
    except OSError:
        pass
print("blocked evidence urls", len(BLOCK), "existing kb urls", len(EXIST))

BAD = re.compile(
    r"/(es|hcp|php|spotlights|weekly|whats-new|statistics|policy|communication-resources|labs?|media|data-research|data"
    r"|pubs|programs|partners|funding|resources?|toolkit|training|nhs-app|ko|zh|vi)(/|$)|sitemap|\.pdf$",
    re.I,
)
CDC = [("covid", 40), ("vaccine-safety", 25), ("vaccines", 30), ("vaccines-children", 15), ("vaccines-adults", 10),
       ("vaccines-pregnancy", 12), ("flu", 14), ("bird-flu", 10), ("ebola", 12), ("marburg", 4), ("measles", 8),
       ("polio", 5), ("hpv", 6), ("rsv", 6), ("dengue", 4), ("malaria", 4)]
NHSQ = [
    (r"/vaccinations/", 28),
    (r"/medicines/(paracetamol-for-adults|ibuprofen-for-adults|aspirin-for-pain-relief|low-dose-aspirin|amoxicillin|metformin"
     r"|atorvastatin|simvastatin|omeprazole|cetirizine|ramipril|amlodipine|prednisolone-tablets-and-liquid|azithromycin|codeine)/?$", 14),
    (r"/pregnancy/keeping-well/", 14),
    (r"/conditions/(covid-19|flu|measles|chickenpox|mumps|rubella|whooping-cough|polio|shingles|dengue|malaria|tuberculosis-tb"
     r"|hepatitis-b|type-2-diabetes|high-blood-pressure-hypertension|high-cholesterol|iron-deficiency-anaemia|vitamin-d)/?$", 16),
    (r"/conditions/baby/(health|first-aid-and-safety)/", 10),
]
MED = ["covid19vaccines", "covid19", "childhoodimmunization", "adultimmunization", "vaccines", "flu", "measles", "polio",
       "hpv", "malaria", "tuberculosis", "anemia", "diabetestype2", "pregnancyandmedicines", "dietarysupplements",
       "antibiotics", "complementaryandintegrativemedicine", "herbalmedicine"]
sel = []
picked = set()


def add(site, prefix, urls, cap, pub):
    seen = 0
    for u in urls:
        if seen >= cap:
            break
        n = norm(u)
        if n in BLOCK or n in EXIST or n in picked or BAD.search(urlparse(u).path):
            continue
        picked.add(n)
        sel.append((site, u, prefix, pub))
        seen += 1


for t, c in CDC:
    us = sorted([u for u in raw["cdc"] if urlparse(u).path.startswith(f"/{t}/")], key=lambda u: (u.count("/"), u))
    add("cdc", "cdc_", us, c, "CDC")
for pat, c in NHSQ:
    add("nhs", "nhs__", sorted(u for u in raw["nhs"] if re.search(pat, urlparse(u).path)), c, "NHS")
add("who", "who__", [u for u in raw["who"] if u.startswith("https://www.who.int")], 60, "WHO")
pri = re.compile(r"covid|corona|vaccin|polio|measles|pregnan|child|baby|infant|dengue|malaria|cancer|diabet|cure")
v = [u for u in raw["vishvas"] if "/english/health/" in u and not u.rstrip("/").endswith("-2")]
add("vishvas", "vishvas__", sorted(v, key=lambda u: (0 if pri.search(u) else 1, u)), 45, "Vishvas News")
add("medline", "medline_", [f"https://medlineplus.gov/{m}.html" for m in MED], len(MED), "MedlinePlus, National Library of Medicine")
print("selected", len(sel), Counter(s[0] for s in sel))

if SAMPLE:
    first = {}
    for x in sel:
        first.setdefault(x[0], x)
    random.seed(3)
    sel = list(first.values()) + random.sample(sel, 3)

ROOT = {
    "cdc": ["main", "#content"],
    "nhs": ["main", "article"],
    "who": [".sf-detail-body-wrapper", "article", "main"],
    "vishvas": [".view-full", ".vn-content", "article", ".entry-content", "main"],
    "medline": ["#topic-summary", "#mplus-content", "article", "main"],
}
CUT = re.compile(
    r"^(sources?|references?|related (pages|articles|content|topics)|resources|more information|page last reviewed"
    r"|next review due|on this page|view all|print|share|last reviewed|find out more|tags?|also read|read more|see also"
    r"|follow us|subscribe)\b.{0,50}$",
    re.I,
)
STOP = re.compile(r"^(sources?|references?|related pages|related (articles|posts)|page last reviewed|about the author)\b", re.I)
DROP = ("nav,header,footer,aside,script,style,form,button,noscript,svg,iframe,figure,.share,.social,.breadcrumb,"
        ".nhsuk-breadcrumb,.nhsuk-contents-list,.nhsuk-review-date,.cdc-dfe-tags,[role=navigation],.related-posts,.sharedaddy")


def extract(html, site):
    sp = BeautifulSoup(html, "html.parser")
    t = sp.find("h1")
    og = sp.find("meta", property="og:title")
    title = (t.get_text(" ", strip=True) if t else "") or (og.get("content", "") if og else "")
    root = None
    for s in ROOT[site]:
        root = sp.select_one(s)
        if root and len(root.get_text().split()) > 60:
            break
    root = root or sp.body
    for bad in root.select(DROP):
        bad.decompose()
    out = []
    for e in root.find_all(["h2", "h3", "h4", "p", "li", "td"]):
        if e.name in ("li", "td") and e.find(["p", "li", "ul", "ol"]):
            continue
        x = re.sub(r"\s+", " ", e.get_text(" ", strip=True)).strip()
        if x == "Related" or "fight against fake news" in x or "free from the ropes" in x or x.startswith("Know the truth!"):
            break
        if not x or x == title or CUT.match(x) or x.startswith("< Go back") or re.match(r"^(Claim Review|Claimed By|Fact Check)\s*:", x):
            continue
        if e.name in ("h2", "h3", "h4") and STOP.match(x):
            break
        out.append(("- " + x) if e.name == "li" else x)
    return title, "\n".join(dict.fromkeys(out))


def slug(u):
    skip = {"index", "english", "health", "news-room", "detail", "fact-sheets", "emergencies", "diseases", "novel-coronavirus-2019"}
    p = [x for x in urlparse(u).path.lower().replace(".html", "").split("/") if x and x not in skip]
    return re.sub(r"[^a-z0-9-]+", "-", "-".join(p))[:85].strip("-") or "page"


LABEL = {"cdc": "CDC", "nhs": "NHS (Open Government Licence v3.0)", "who": "WHO",
         "vishvas": "Vishvas News — Health Fact Check", "medline": "MedlinePlus, National Library of Medicine"}
csvp = os.path.join(KB, "SOURCES.csv")
done = set()
if os.path.exists(csvp):
    done = {r["url"] for r in csv.DictReader(open(csvp, encoding="utf-8"))}

for i, (site, u, pre, pub) in enumerate(sel):
    if u in done:
        continue
    time.sleep(random.uniform(1.5, 3.0))
    try:
        r = requests.get(u, headers=H, timeout=40)
    except Exception as e:
        print(i, "ERR", u, str(e)[:80])
        continue
    if r.status_code != 200:
        print(i, r.status_code, u)
        continue
    r.encoding = "utf-8"
    if norm(r.url) in BLOCK or norm(r.url) in EXIST:
        print(i, "redirect-to-blocked-or-existing", r.url)
        continue
    title, body = extract(r.text, site)
    w = len(body.split())
    if w < 150:
        print(i, "short", w, u)
        continue
    fn = pre + slug(r.url) + ".txt"
    txt = f"SOURCE: {LABEL[site]}\nTITLE: {title}\nURL: {r.url}\n{'─' * 60}\n\n{body}\n"
    if SAMPLE:
        print("=" * 30, fn, w)
        print(txt[:900])
        print("... END:", body[-220:])
        continue
    open(os.path.join(KB, fn), "w", encoding="utf-8", newline="\n").write(txt)
    new = not os.path.exists(csvp)
    with open(csvp, "a", encoding="utf-8", newline="") as f:
        c = csv.writer(f, lineterminator="\n")
        if new:
            c.writerow(["filename", "publisher", "url", "date_accessed"])
        c.writerow([fn, pub, r.url, TODAY])
    print(i, "ok", w, fn)
print("DONE")
