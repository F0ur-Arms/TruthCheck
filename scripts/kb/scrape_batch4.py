"""KB batch 4: U.S. government consumer health pages (public domain).

MedlinePlus health topics (NLM's daily XML release, English full summaries) and NCCIH "Herbs at a
Glance" pages (turmeric, ashwagandha, fenugreek, cinnamon and other remedies that Indian health
forwards often mention). Both are general reference pages written for the public, not fact-checks of
specific claims, so adding them does not leak benchmark answers. Topics already in the knowledge base
are skipped. The NIH Office of Dietary Supplements site refuses automated requests, so it is not used.

usage: python scripts/kb/scrape_batch4.py [CACHE_DIR]
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
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
KB = "data/medical_kb"
CACHE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/truthcheck_kb4"
os.makedirs(CACHE, exist_ok=True)
UA = "Mozilla/5.0 (compatible; TruthCheckResearchBot/0.1; academic health misinformation project)"
TODAY = datetime.date.today().isoformat()
RULE = "─" * 60
MIN_WORDS = 60
NCCIH_SKIP = {"atoz", "espanol", "herblist-app", "herbsataglance", "know-science", "pain", "providers",
              "safety", "tips", "complementary-alternative-or-integrative-health-whats-in-a-name"}
NCCIH_STOP = re.compile(r"^(For More Information|Key References|Selected References|Other References|"
                        r"Acknowledgments|This publication is not copyrighted|NCCIH Clearinghouse)\b")

done = {r["filename"] for r in csv.DictReader(open(os.path.join(KB, "SOURCES.csv"), encoding="utf-8"))}
done |= set(os.listdir(KB))


def get(url, delay=True):
    if delay:
        time.sleep(random.uniform(1.0, 2.0))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def html_to_text(fragment):
    fragment = re.sub(r"<(script|style|figure|figcaption)\b.*?</\1>", "", fragment, flags=re.S | re.I)
    fragment = re.sub(r"<(h[1-6]|p|li|div|br|tr)\b[^>]*>", "\n", fragment, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", "", fragment)).replace("\xa0", " ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


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


def medlineplus():
    index = get("https://medlineplus.gov/xml.html", delay=False).decode("utf-8", "replace")
    name = sorted(set(re.findall(r"mplus_topics_(\d{4}-\d{2}-\d{2})\.xml", index)))[-1]
    path = os.path.join(CACHE, f"mplus_topics_{name}.xml")
    if not os.path.exists(path):
        open(path, "wb").write(get(f"https://medlineplus.gov/xml/mplus_topics_{name}.xml", delay=False))
    added = 0
    for topic in ET.parse(path).getroot().iter("health-topic"):
        if topic.get("language") != "English":
            continue
        url, title = topic.get("url", ""), topic.get("title", "")
        summary = topic.findtext("full-summary") or ""
        slug = re.sub(r"[^a-z0-9]+", "", url.rsplit("/", 1)[-1].removesuffix(".html").lower())
        also = [a.text for a in topic.findall("also-called") if a.text]
        body = html_to_text(summary)
        if also:
            body = f"Also called: {', '.join(also)}\n{body}"
        added += write(f"medline_{slug}.txt", "MedlinePlus, U.S. National Library of Medicine (public domain)",
                       title, url, "MedlinePlus (NLM)", body)
    print(f"MedlinePlus: {added} topics added")


def nccih():
    listing = get("https://www.nccih.nih.gov/health/herbsataglance", delay=False).decode("utf-8", "replace")
    herbs = sorted(set(re.findall(r'href="/health/([a-z0-9-]+)"', listing)) - NCCIH_SKIP)
    added = 0
    for herb in herbs:
        fn = f"nih_{herb.replace('-', '_')}.txt"
        if fn in done:
            print("skip (already in KB)", fn)
            continue
        url = f"https://www.nccih.nih.gov/health/{herb}"
        try:
            page = get(url).decode("utf-8", "replace")
        except Exception as e:
            print("ERR", url, str(e)[:80])
            continue
        main = re.search(r"<main\b.*?</main>", page, re.S | re.I)
        lines = html_to_text(main.group(0) if main else page).split("\n")
        kept = []
        for line in lines:
            if NCCIH_STOP.match(line):
                break
            if line.startswith("Photo credit") or "Photo credit:" in line:
                line = line.split("Photo credit:")[0].strip()
            if line:
                kept.append(line)
        # The page <title> is the site name; the herb name is the first line of the article.
        title = kept[0] if kept else herb
        added += write(fn, "NCCIH, U.S. National Institutes of Health (public domain)", title, url,
                       "NCCIH (NIH)", "\n".join(kept))
    print(f"NCCIH: {added} herb pages added")


if __name__ == "__main__":
    medlineplus()
    nccih()
