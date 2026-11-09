"""KB batch 5: alternative-treatment and complementary-health references (U.S. government, public domain).

- NCI PDQ patient summaries on complementary and alternative cancer treatments (laetrile, Gerson,
  714-X, antineoplastons, Essiac, mistletoe, cannabis, ...) and NCI's "Common Cancer Myths" page.
  "X treats cancer" is the largest group of claims the knowledge base could not answer in round 13.
- NCCIH A-Z health topics that are not herbs (Ayurveda, homeopathy, chelation, magnets, detoxes,
  colloidal silver, ...); the herb pages came in batch 4.
These are general reference pages written for the public, not fact-checks of specific claims.
NCI files use the nih_nci_ prefix so they get the NIH source tier (NCI is part of NIH).

usage: python scripts/kb/scrape_batch5.py
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

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
KB = "data/medical_kb"
UA = "Mozilla/5.0 (compatible; TruthCheckResearchBot/0.1; academic health misinformation project)"
TODAY = datetime.date.today().isoformat()
RULE = "─" * 60
MIN_WORDS = 60
NCI = "https://www.cancer.gov"
NCCIH = "https://www.nccih.nih.gov"
# Navigation, funding and how-to pages rather than health information.
NCCIH_SKIP = re.compile(r"^(atoz|espanol|providers|tips|safety|know-science|herbsataglance|herblist-app|pain|"
                        r"credentialing|finding-information|how-to-find|paying-for|in-the-news|be-an-informed|"
                        r"complementary-alternative-or-integrative|are-you-considering|pediatrics|mens-health|"
                        r"mental-health|mind-and-body-practices|dietary-and-herbal-supplements)")
NCCIH_STOP = re.compile(r"^(For More Information|Key References|Selected References|Other References|"
                        r"Acknowledgments|This publication is not copyrighted|NCCIH Clearinghouse)\b")

done = {r["filename"] for r in csv.DictReader(open(os.path.join(KB, "SOURCES.csv"), encoding="utf-8"))}
done |= set(os.listdir(KB))


def get(url):
    time.sleep(random.uniform(1.0, 2.0))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode("utf-8", "replace")


def lines_of(page):
    main = re.search(r"<main\b.*?</main>", page, re.S | re.I)
    frag = re.sub(r"<(script|style|figure|figcaption|nav)\b.*?</\1>", "", main.group(0) if main else page,
                  flags=re.S | re.I)
    frag = re.sub(r"<(h[1-6]|p|li|div|br|tr)\b[^>]*>", "\n", frag, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", "", frag)).replace("\xa0", " ")
    return [l for l in (re.sub(r"\s+", " ", x).strip() for x in text.split("\n")) if l]


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


def nci():
    listing = get(f"{NCI}/about-cancer/treatment/cam/patient")
    slugs = sorted(set(re.findall(r'href="/about-cancer/treatment/cam/patient/([a-z0-9-]+)"', listing)))
    pages = [(f"/about-cancer/treatment/cam/patient/{s}", s.removesuffix("-pdq")) for s in slugs]
    pages.append(("/about-cancer/causes-prevention/risk/myths", "cancer-myths"))
    added = 0
    for path, slug in pages:
        url = NCI + path
        try:
            lines = lines_of(get(url))
        except Exception as e:
            print("ERR", url, str(e)[:80])
            continue
        title = lines[0] if lines else slug
        # PDQ pages open with a table of contents; the text starts after the professional-version link
        # and ends at the "About This PDQ Summary" boilerplate.
        start = next((i + 1 for i, l in enumerate(lines) if l == "Go to Health Professional Version"), 1)
        end = next((i for i, l in enumerate(lines) if i > start and l == "About This PDQ Summary"), len(lines))
        body = "\n".join(lines[start:end])
        added += write(f"nih_nci_{slug.replace('-', '_')}.txt", "National Cancer Institute, NIH (public domain)",
                       title, url, "NCI (NIH)", body)
    print(f"NCI: {added} pages added")


def nccih_topics():
    listing = get(f"{NCCIH}/health/atoz")
    slugs = sorted(set(re.findall(r'href="/health/([a-z0-9-]+)"', listing)))
    added = 0
    for slug in slugs:
        if NCCIH_SKIP.match(slug):
            continue
        fn = f"nih_{slug.replace('-', '_')}.txt"
        if fn in done:
            continue
        url = f"{NCCIH}/health/{slug}"
        try:
            lines = lines_of(get(url))
        except Exception as e:
            print("ERR", url, str(e)[:80])
            continue
        kept = []
        for line in lines:
            if NCCIH_STOP.match(line):
                break
            kept.append(line.split("Photo credit:")[0].strip() or line)
        added += write(fn, "NCCIH, U.S. National Institutes of Health (public domain)", kept[0] if kept else slug,
                       url, "NCCIH (NIH)", "\n".join(kept))
    print(f"NCCIH topics: {added} pages added")


if __name__ == "__main__":
    nci()
    nccih_topics()
