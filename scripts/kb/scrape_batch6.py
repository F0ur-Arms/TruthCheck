"""KB batch 6: public-domain U.S. government (NIH) pages, aimed at the topics
with the most unanswered benchmark claims in round 19 (cancer causes, drugs, mental health, COVID-19).

- NCI "Cancer causes and prevention": risk-factor pages and their fact sheets (diet, artificial
  sweeteners, alcohol, radiation, cell phones, hormones, infections, substances, ...).
- NIMH health topics (mental health) and NIDA research topics (cannabis, vaping, opioids, ...).
- (WHO's COVID-19 mythbusters page was tried and dropped: it holds three HealthVer benchmark claims
  word for word, so the system could look those answers up.)
These are general reference pages, not fact-checks of specific benchmark claims; run
scripts/kb/check_kb.py on the new files before committing. NCI, NIMH and NIDA files use the nih_
prefix (NIH institutes) and the WHO file the who__ prefix, so they get those source tiers.

usage: python scripts/kb/scrape_batch6.py
needs: standard library only
"""
import csv
import datetime
import html
import os
import random
import re
import sys
import time
import urllib.request
from urllib.parse import urljoin, urlparse

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
KB = "data/medical_kb"
UA = "Mozilla/5.0 (compatible; TruthCheckResearchBot/0.1; academic health misinformation project)"
TODAY = datetime.date.today().isoformat()
RULE = "─" * 60
MIN_WORDS = 80
STOP = re.compile(r"^(Selected References|References|Related Resources|Related Content|Share this page|"
                  r"Last Reviewed|Reprints|Clinical Trials|For More Information|Additional Resources|"
                  r"If you would like to reproduce|This content is provided by|Contact Us|Sign up)\b", re.I)

# (publisher, label, start URL, path prefix to stay inside, filename prefix, max pages, crawl depth)
SITES = [
    ("NCI (NIH)", "National Cancer Institute, NIH (public domain)",
     "https://www.cancer.gov/about-cancer/causes-prevention/risk", "/about-cancer/causes-prevention/",
     "nih_nci_risk_", 160, 2),
    ("NIMH (NIH)", "National Institute of Mental Health, NIH (public domain)",
     "https://www.nimh.nih.gov/health/topics", "/health/topics/", "nih_nimh_", 70, 1),
    ("NIDA (NIH)", "National Institute on Drug Abuse, NIH (public domain)",
     "https://nida.nih.gov/research-topics", "/research-topics/", "nih_nida_", 60, 1),
]
WHO_MYTHS = "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters"

done = {r["filename"] for r in csv.DictReader(open(os.path.join(KB, "SOURCES.csv"), encoding="utf-8"))}
done |= set(os.listdir(KB))


def get(url):
    time.sleep(random.uniform(1.0, 2.0))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode("utf-8", "replace")


def lines_of(page):
    main = re.search(r"<main\b.*?</main>", page, re.S | re.I) or re.search(r"<article\b.*?</article>", page, re.S | re.I)
    frag = re.sub(r"<(script|style|figure|figcaption|nav|aside|footer)\b.*?</\1>", "", main.group(0) if main else page,
                  flags=re.S | re.I)
    frag = re.sub(r"<(h[1-6]|p|li|div|br|tr|dt|dd)\b[^>]*>", "\n", frag, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", "", frag)).replace("\xa0", " ")
    return [l for l in (re.sub(r"\s+", " ", x).strip() for x in text.split("\n")) if l]


def body_of(lines):
    kept = []
    for line in lines:
        if STOP.match(line):
            break
        if line not in ("On This Page", "Print", "Email", "Share"):
            kept.append(line)
    return kept


def write(fn, label, title, url, publisher, body):
    if fn in done:
        print("skip (already in KB)", fn)
        return False
    if len(body.split()) < MIN_WORDS:
        print("skip (too short)", fn)
        return False
    done.add(fn)
    with open(os.path.join(KB, fn), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"SOURCE: {label}\nTITLE: {title}\nURL: {url}\n{RULE}\n\n{body}\n")
    with open(os.path.join(KB, "SOURCES.csv"), "a", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow([fn, publisher, url, TODAY])
    print("ok", len(body.split()), fn)
    return True


def crawl(publisher, label, start, prefix, fn_prefix, limit, depth):
    host = f"{urlparse(start).scheme}://{urlparse(start).netloc}"
    seen, frontier, added = {start}, [(start, 0)], 0
    while frontier and len(seen) <= limit:
        url, d = frontier.pop(0)
        try:
            page = get(url)
        except Exception as e:
            print("ERR", url, str(e)[:80])
            continue
        if d < depth:
            for href in re.findall(r'href="([^"#?]+)"', page):
                link = urljoin(host, href).rstrip("/")
                if urlparse(link).netloc == urlparse(host).netloc and urlparse(link).path.startswith(prefix) \
                        and not re.search(r"\.(pdf|jpg|png|zip)$|/(espanol|es|spanish)\b", link) and link not in seen:
                    seen.add(link)
                    frontier.append((link, d + 1))
        if url == start:
            continue  # index pages are link lists
        lines = body_of(lines_of(page))
        if not lines:
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", urlparse(url).path[len(prefix):].lower()).strip("_")[:90]
        added += write(f"{fn_prefix}{slug}.txt", label, lines[0], url, publisher, "\n".join(lines))
    print(f"{publisher}: {added} pages added")


def who_mythbusters():
    lines = body_of(lines_of(get(WHO_MYTHS)))
    write("who__covid19_mythbusters.txt", "World Health Organization", "COVID-19 advice for the public: Mythbusters",
          WHO_MYTHS, "WHO", "\n".join(lines))


if __name__ == "__main__":
    for site in SITES:
        crawl(*site)
    # who_mythbusters() is not run: see the note at the top.
