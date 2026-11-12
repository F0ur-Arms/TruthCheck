"""Collect Hindi health fact-checks from Indian fact-checkers as golden-set candidates.

Sources:
  Vishvas News (Hindi)    health section + site search      (ClaimReview markup)
  Alt News (Hindi)        WordPress search API               (ClaimReview markup)
  Newschecker (Hindi)     WordPress search API + categories  ("Claim/Fact/Result" layout)

Searches use Hindi health keywords because many health claims are filed under
"viral" or "society" rather than a health section. Writes one JSON object per
fact-check. These are candidates only: each one is reviewed by hand before it
goes into golden_v1.csv.
"""

import argparse
import html
import json
import re
import subprocess
import time
import urllib.parse

DELAY = 0.4
KEYWORDS = [
    "इलाज", "कैंसर", "वैक्सीन", "टीका", "डायबिटीज", "शुगर", "डेंगू", "कोरोना", "दवा", "बीमारी",
    "हार्ट अटैक", "ब्लड प्रेशर", "किडनी", "स्वास्थ्य", "डॉक्टर", "आयुर्वेद", "घरेलू उपाय", "नुस्खा",
    "वायरस", "मलेरिया", "पैरासिटामोल", "गर्भवती", "बच्चों", "मोटापा", "संक्रमण",
]
NEWSCHECKER_CATEGORIES = [5983, 5963, 6026]
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def fetch(url):
    # curl follows 308 redirects, which urllib on Python 3.9 does not.
    result = subprocess.run(["curl", "-sL", "-m", "30", "-A", "Mozilla/5.0", url], capture_output=True)
    return result.stdout.decode("utf-8", "replace")


def clean(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def claim_reviews(page):
    out = []
    for block in re.findall(r"<script[^>]*ld\+json[^>]*>(.*?)</script>", page, re.S):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                if node.get("@type") == "ClaimReview":
                    out.append(node)
                stack.extend(v for v in node.values() if isinstance(v, (list, dict)))
    return out


def conclusion(text):
    match = re.search(r"निष्कर्ष\s*:?\s*(.{20,600}?)(?:पूरा सच जानें|सब को बताएं|Claim Review|Result|$)", text)
    return clean(match.group(1)) if match else ""


def from_claimreview_page(url, publisher):
    page = fetch(url)
    reviews = claim_reviews(page)
    if not reviews:
        return None
    review = reviews[0]
    claim = clean(review.get("claimReviewed"))
    if not DEVANAGARI.search(claim):
        return None
    title = re.search(r'<meta property="og:title" content="([^"]*)', page)
    return {
        "claim_text": claim,
        "rating": clean((review.get("reviewRating") or {}).get("alternateName")),
        "publisher": publisher,
        "url": url,
        "date": (review.get("datePublished") or "")[:10],
        "headline": clean(title.group(1)) if title else "",
        "evidence_quote": conclusion(clean(page)),
    }


def vishvas_links(max_pages):
    links = []
    article = r'href="(https://www.vishvasnews.com/(?!author|tag|category|english|urdu|punjabi|marathi|' \
              r'assamese|odia|gujarati|bangla|telugu|tamil|malayalam)[a-z-]+/[a-z0-9-]+/)"'
    for n in range(1, max_pages + 1):
        page = fetch("https://www.vishvasnews.com/health/" + (f"page/{n}/" if n > 1 else ""))
        found = re.findall(article, page)
        if not found:
            break
        links += found
    for word in KEYWORDS:
        q = urllib.parse.quote(word)
        for n in range(1, 6):
            page = fetch(f"https://www.vishvasnews.com/{f'page/{n}/' if n > 1 else ''}?s={q}")
            found = re.findall(article, page)
            if not found:
                break
            links += found
            time.sleep(DELAY)
    return list(dict.fromkeys(links))


def wp_search_links(base, per_keyword):
    links = []
    for word in KEYWORDS:
        q = urllib.parse.quote(word)
        try:
            posts = json.loads(fetch(f"{base}/wp-json/wp/v2/posts?search={q}&per_page={per_keyword}&_fields=link"))
        except json.JSONDecodeError:
            continue
        links += [p["link"] for p in posts if isinstance(p, dict)]
        time.sleep(DELAY)
    return list(dict.fromkeys(links))


def newschecker_posts(per_keyword):
    base = "https://newschecker.in/hi/wp-json/wp/v2/posts?_fields=link,date,title,content&per_page=50"
    urls = [f"{base}&search={urllib.parse.quote(w)}" for w in KEYWORDS]
    urls += [f"{base}&categories={c}&page={n}" for c in NEWSCHECKER_CATEGORIES for n in range(1, 5)]
    seen = set()
    for url in urls:
        try:
            posts = json.loads(fetch(url))
        except json.JSONDecodeError:
            continue
        for post in posts if isinstance(posts, list) else []:
            if post["link"] in seen:
                continue
            seen.add(post["link"])
            yield post
        time.sleep(DELAY)


def scrape_newschecker():
    for post in newschecker_posts(50):
        text = clean(post["content"]["rendered"])
        claim = (re.search(r"\bClaim\b\s*:?\s*(.{10,400}?)\s*\bFact\b", text)
                 or re.search(r"Claim Review\s*:\s*(.{10,400}?)\s*Claimed By", text))
        result = (re.search(r"\bResult\b\s*:?\s*([A-Za-z ]{3,30}?)\s*(?:Sources|Our Sources|$)", text)
                  or re.search(r"Fact Check\s*:\s*([A-Za-z ]{3,30}?)\s*(?:Sources|Our Sources|$|\W)", text))
        if not claim or not result or not DEVANAGARI.search(claim.group(1)):
            continue
        yield {
            "claim_text": claim.group(1).strip(),
            "rating": result.group(1).strip(),
            "publisher": "newschecker.in",
            "url": post["link"],
            "date": post["date"][:10],
            "headline": clean(post["title"]["rendered"]),
            "evidence_quote": conclusion(text),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--skip-urls", help="file of URLs already in the golden set, one per line")
    parser.add_argument("--vishvas-pages", type=int, default=40)
    args = parser.parse_args()
    skip = set()
    if args.skip_urls:
        skip = {line.strip().rstrip("/") for line in open(args.skip_urls, encoding="utf-8") if line.strip()}

    count = 0
    with open(args.output, "w", encoding="utf-8") as out:
        def emit(record):
            nonlocal count
            if record and record["url"].rstrip("/") not in skip:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                count += 1

        for record in scrape_newschecker():
            emit(record)
        for link in wp_search_links("https://www.altnews.in/hindi", 30):
            if link.rstrip("/") not in skip:
                emit(from_claimreview_page(link, "altnews.in"))
                time.sleep(DELAY)
        for link in vishvas_links(args.vishvas_pages):
            if link.rstrip("/") not in skip:
                emit(from_claimreview_page(link, "vishvasnews.com"))
                time.sleep(DELAY)
    print(f"Wrote {count} candidates to {args.output}")


if __name__ == "__main__":
    main()
