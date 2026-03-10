"""
TruthCheck — Vishvas News Health Fact-Check Scraper
=====================================================
Scrapes Indian health misinformation fact-checks from Vishvas News.
These are gold for your RAG — each article debunks a specific viral claim
(cancer cures, COVID myths, vaccine misinformation, food myths etc.)

Install:
    pip install requests beautifulsoup4 trafilatura

Run:
    python scrape_vishvasnews.py

Output:
    C:\\Users\\Shivam Kumar\\frenemy\\TruthCheck\\data\\medical_kb\\vishvas__*.txt
"""

import os
import re
import time
import random
import requests
import trafilatura
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = r"C:\Users\Shivam Kumar\frenemy\TruthCheck\data\medical_kb"
MIN_WORDS  = 150
DELAY_MIN  = 1.5
DELAY_MAX  = 3.0

HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer":         "https://www.vishvasnews.com/english/health/",
}

# ─────────────────────────────────────────────────────────────────────────────
# ARTICLE URLs — 100 Vishvas News health fact-checks
# Add more by pasting below the last entry
# ─────────────────────────────────────────────────────────────────────────────

ARTICLE_URLS = [
    "https://www.vishvasnews.com/english/health/fact-check-can-lime-cure-breast-lumps-know-from-experts/",
    "https://www.vishvasnews.com/english/health/fact-check-claim-that-eating-turmeric-lemon-basil-and-neem-cures-stage-4-cancer-is-false/",
    "https://www.vishvasnews.com/english/health/fact-check-the-claim-that-drinking-fenugreek-clove-cardamom-and-cinnamon-mixed-in-water-cures-last-stage-cancer-is-false/",
    "https://www.vishvasnews.com/english/health/fact-check-viral-post-claiming-toothpaste-causes-cancer-is-fake-sodium-lauryl-sulphate-sls-is-not-carcinogenic-say-experts/",
    "https://www.vishvasnews.com/english/health/fact-check-claim-of-excessive-usage-of-earbuds-earphones-could-cause-facial-paralysis-is-false/",
    "https://www.vishvasnews.com/english/health/fact-check-conspiracy-theory-about-covid-19-vaccines-and-sterility-is-viral-again-2009-video-has-false-information/",
    "https://www.vishvasnews.com/english/health/fact-check-viral-post-on-covid-xbb-variant-is-misleading/",
    "https://www.vishvasnews.com/english/health/fact-check-no-medical-evidence-that-airpods-cause-brain-tumours-as-viral-post-claims/",
    "https://www.vishvasnews.com/english/health/fact-check-this-breathing-test-doesnt-measure-lung-health-or-oxygen-levels-viral-video-is-misleading/",
    "https://www.vishvasnews.com/english/health/fact-check-fabricated-advisory-on-monkeypox-outbreak-attributed-to-who/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-garlic-is-remedy-for-ear-infections-headache-is-misleading-experts-dismissed-the-viral-claim/",
    "https://www.vishvasnews.com/english/health/fact-check-the-man-in-this-video-isnt-dr-sushil-razdan-covid-19-can-not-be-prevented-by-snorting-dry-ginger-powder/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-cupping-can-remove-covid-19-vaccine-content-from-the-body-is-fake/",
    "https://www.vishvasnews.com/english/health/quick-fact-check-fake-post-claiming-anaesthetics-can-kill-coronavirus-resurfaces/",
    "https://www.vishvasnews.com/english/health/fact-check-remdesivir-injection-is-not-available-on-pradhan-mantri-jan-aushadhi-kendras-viral-post-is-misleading/",
    "https://www.vishvasnews.com/english/health/quick-fact-check-fake-post-claiming-who-listed-out-7-brain-damaging-habits-resurfaces/",
    "https://www.vishvasnews.com/english/health/fact-check-no-face-masks-alone-cannot-help-in-preventing-covid-19-viral-post-is-missing-crucial-context/",
    "https://www.vishvasnews.com/english/health/quick-fact-check-misleading-post-claiming-black-garlic-can-cure-cancer-resurfaces/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-measles-vaccine-wont-work-for-people-born-before-1989-is-misleading/",
    "https://www.vishvasnews.com/english/health/fact-check-measures-mentioned-in-the-video-are-not-a-cure-for-coronavirus-but-tips-to-increase-immunity/",
    "https://www.vishvasnews.com/english/health/fact-check-the-doctor-in-the-viral-claim-arrested-in-kidney-racket-not-for-giving-fake-covid-positive-reports/",
    "https://www.vishvasnews.com/english/health/fact-check-no-evidence-proves-coronavirus-remains-in-throat-for-4-days/",
    "https://www.vishvasnews.com/english/health/fact-check-this-viral-message-on-novel-coronavirus-is-not-issued-by-unicef/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-china-seeks-courts-approval-to-kill-20000-coronavirus-patients-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-no-dettol-didnt-test-the-2019-novel-coronavirus-strains-misleading-post-getting-viral/",
    "https://www.vishvasnews.com/english/health/fact-check-fake-post-on-paracetamol-p-500-containing-machupo-virus-resurfaces/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-coronavirus-discovered-in-broilers-chicken-in-recent-outbreak-is-fake/",
    "https://www.vishvasnews.com/english/health/world-cancer-day-2020-debunking-most-shared-cancer-myths-and-misconceptions/",
    "https://www.vishvasnews.com/english/health/fact-check-fake-post-on-coronavirus-outbreak-attributed-to-health-ministry-advisory-getting-viral/",
    "https://www.vishvasnews.com/english/health/fact-check-misinformation-related-to-coronavirus-outbreak-getting-viral/",
    "https://www.vishvasnews.com/english/health/fact-check-the-post-claiming-that-a-woman-delivered-17-babies-at-once-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-that-consuming-turmeric-juice-cow-urine-in-certain-proportions-can-cure-cancer-completely-is-fake/",
    "https://www.vishvasnews.com/english/health/quick-fact-check-fake-post-on-spotting-a-camel-in-image-to-detect-alzheimers-resurfaces/",
    "https://www.vishvasnews.com/english/health/quick-fact-check-fake-post-on-image-determining-the-state-of-mind-resurfaces/",
    "https://www.vishvasnews.com/english/health/fact-check-can-consuming-wax-coated-apples-kill-you-misleading-video-getting-viral/",
    "https://www.vishvasnews.com/english/health/quick-fact-check-fake-post-claiming-harmful-tablets-contain-tissue-paper-resurfaces/",
    "https://www.vishvasnews.com/english/health/fact-check-video-claiming-eating-bitter-guard-or-radish-over-ladyfinger-is-harmful-is-fake-as-per-doctors/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-combiflam-can-kill-you-is-misleading/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-tongue-exercise-cures-alzheimers-and-other-illnesses-is-misleading/",
    "https://www.vishvasnews.com/english/health/fact-check-giving-birth-cant-be-measured-in-del-pain-is-subjective-fake-post-getting-viral/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-wild-aubergine-can-cure-any-type-of-cancer-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-does-the-5-second-rule-really-exist-it-is-a-myth/",
    "https://www.vishvasnews.com/english/health/fact-check-the-post-claiming-that-baking-soda-can-cure-cancer-is-fake/",
    "https://www.vishvasnews.com/english/health/quick-fact-check-misleading-post-claiming-sprouted-wheat-can-cure-diabetes-resurfaces/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-that-a-smoothie-can-cure-diabetes-is-misleading/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-that-burning-camphor-and-inhaling-the-smell-combats-air-pollution-is-misleading/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-101-year-old-woman-delivered-a-baby-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-no-this-home-made-mixture-cannot-cure-cancer-in-3-days/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-that-bee-venom-treatment-heals-infection-is-misleading/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-children-recently-vaccinated-with-polio-vaccine-spreads-virus-causing-hfmd-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-no-up-police-did-not-issue-any-aids-related-warning-the-viral-post-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-this-homeopathy-medicine-can-cure-dengue-within-48-hours-is-a-misleading-claim/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-that-drinking-cold-water-after-meals-can-cause-cancer-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-that-nestle-accepts-adding-juice-of-beef-in-kitkat-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-flu-shots-make-people-contagious-is-a-misleading-post/",
    "https://www.vishvasnews.com/english/health/fact-check-no-cream-of-tartar-mixed-with-orange-juice-wont-help-you-quit-smoking/",
    "https://www.vishvasnews.com/english/health/fact-check-the-post-claiming-that-diabetes-can-be-cured-by-consuming-this-decoction-is-misleading/",
    "https://www.vishvasnews.com/english/health/fact-check-viral-post-claiming-cancer-is-not-disease-but-fungus-is-misleading/",
    "https://www.vishvasnews.com/english/health/fact-check-no-a-camel-in-an-image-can-not-help-in-diagnosing-alzheimers/",
    "https://www.vishvasnews.com/english/health/fact-check-no-miracle-mineral-solution-cannot-cure-diseases-such-as-autism-or-depression/",
    "https://www.vishvasnews.com/english/health/fact-check-pricking-needles-on-fingertips-or-pulling-ears-of-the-stroke-victim-can-save-his-life-is-a-misleading-post/",
    "https://www.vishvasnews.com/english/health/fact-check-photoshopped-image-of-a-chest-x-ray-going-viral-with-a-bizarre-narrative/",
    "https://www.vishvasnews.com/english/health/fact-check-no-the-new-born-baby-is-not-105-years-old-she-is-suffering-from-a-skin-disease/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-sprouted-wheat-can-cure-diabetes-and-prevent-heart-attack-is-misleading/",
    "https://www.vishvasnews.com/english/health/fact-check-sucking-cancer-by-mouth-false-claims-by-a-baba-going-viral/",
    "https://www.vishvasnews.com/english/health/fact-check-ginger-is-10000-times-more-effective-than-chemotherapy-is-a-misleading-claim/",
    "https://www.vishvasnews.com/english/health/fact-check-pepaero-cures-dengue-in-48-hours-is-a-misleading-claim/",
    "https://www.vishvasnews.com/english/health/fact-check-okra-soaked-water-cannot-cure-diabetes/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-noodles-being-surgically-removed-from-intestines-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-slaked-lime-cures-70-diseases-is-misleading/",
    "https://www.vishvasnews.com/english/health/fact-check-viral-post-regarding-contraceptive-pills-is-misleading/",
    "https://www.vishvasnews.com/english/health/fact-check-no-a-drowned-person-cannot-revive-if-covered-in-salt/",
    "https://www.vishvasnews.com/english/health/fact-check-no-this-exercise-cannot-cure-heart-blockage-in-7-days/",
    "https://www.vishvasnews.com/english/health/fact-check-no-knuckle-cracking-does-not-cause-arthritis/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-caripill-cures-dengue-in-48-hours-is-misleading/",
    "https://www.vishvasnews.com/english/health/fact-check-no-paracetamol-does-not-remain-in-the-body-for-5-years/",
    "https://www.vishvasnews.com/english/health/fact-check-no-mr-vaccine-does-not-cause-infertility-the-viral-post-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-deodorants-cause-breast-cancer-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-the-red-dot-eye-test-to-check-your-vision-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-two-handfuls-of-cashews-equivalent-to-a-prozac-dose-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-no-unesco-did-not-declare-idly-as-the-healthiest-breakfast-of-the-world/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-coconut-oil-can-prevent-dengue-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-cheap-and-harmful-tablets-contain-tissue-paper-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-lemons-are-10000-times-stronger-than-chemotherapy-is-a-fake-claim/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-mother-killed-children-by-mixing-cough-syrup-with-milk-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-onions-absorb-bacteria-to-prevent-flu-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-calling-a-sleeping-persons-name-can-cause-brain-damage-is-a-fake-post/",
    "https://www.vishvasnews.com/english/health/fact-check-false-claims-of-curing-cancer-getting-viral/",
    "https://www.vishvasnews.com/english/health/fact-check-no-microwave-ovens-are-not-banned-in-japan/",
    "https://www.vishvasnews.com/english/health/fact-check-smallpox-treatment-mr-vac-is-a-slow-poison-for-children-is-a-fake-post/",
    "https://www.vishvasnews.com/english/health/fact-check-all-deadly-poisons-can-be-treated-immediately-on-call-is-a-fake-post/",
    "https://www.vishvasnews.com/english/health/fact-check-applying-black-tea-on-hair-can-turn-white-hair-black-is-a-misleading-post/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-magical-chutney-can-cure-diabetes-completely-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-boy-infected-with-hiv-from-tainted-pineapple-is-a-fake-news/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-cadbury-products-contaminated-with-hiv-positive-blood-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-cold-drink-and-mangoes-is-a-deadly-combination-is-a-fake-post/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-consuming-banana-and-egg-is-deadly-combination-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-post-showing-bananas-injected-with-hiv-blood-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-hot-coconut-water-can-kill-cancer-cells-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-hba1c-between-7-to-8-is-normal-is-misleading/",
    "https://www.vishvasnews.com/english/health/fact-check-post-claiming-uni-sto-medicine-can-cure-stones-in-4-hours-is-fake/",
    "https://www.vishvasnews.com/english/health/fact-check-no-marijuana-does-not-contain-alien-dna-from-outside-of-our-solar-system/",
    "https://www.vishvasnews.com/english/health/fact-check-these-are-not-photos-of-a-man-with-an-artificial-heart/",
    # ── Paste more URLs below ─────────────────────────────────────────────────
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def url_to_slug(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', slug)[:80]


def already_saved(slug: str) -> bool:
    return os.path.exists(os.path.join(OUTPUT_DIR, f"vishvas__{slug}.txt"))


def clean_text(raw: str) -> str:
    lines = raw.splitlines()
    kept = []
    for line in lines:
        line = line.strip()
        if len(line) < 35:
            continue
        if re.search(
            r'(subscribe|newsletter|©|cookie|follow us|share this|'
             r'also read|read more|vishvas news|fact.check rating|'
             r'conclusion:|result:|fake|misleading|false)',   # keep these — they ARE the verdict
            line, re.I
        ):
            # Don't skip verdict lines — they're the most useful for RAG
            if re.search(r'(conclusion|result|verdict|claim|fact.check)', line, re.I):
                kept.append(line)
            continue
        kept.append(line)
    return re.sub(r'\n{3,}', '\n\n', "\n".join(kept)).strip()


def fetch_and_extract(url: str, session: requests.Session) -> tuple[str, str]:
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 403:
            return "", "BLOCKED"
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        return "", f"ERROR:{e}"

    # Extract title
    title = ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else url_to_slug(url).replace("-", " ").title()
    except Exception:
        pass

    # Try trafilatura first
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
        favor_precision=True,
    )

    # Fallback: target Vishvas News article body
    if not extracted:
        try:
            soup = BeautifulSoup(html, "html.parser")
            body = (
                soup.find("div", class_=re.compile(r'article[_\-]?content|post[_\-]?content|entry[_\-]?content|story', re.I))
                or soup.find("div", {"itemprop": "articleBody"})
                or soup.find("article")
            )
            if body:
                for tag in body(["script", "style", "nav", "aside", "figure", "button"]):
                    tag.decompose()
                extracted = body.get_text(separator="\n")
        except Exception:
            pass

    if not extracted:
        return title, ""

    return title, clean_text(extracted)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Filter out the bare anchor link (#) that slipped in
    valid_urls = [u for u in ARTICLE_URLS if u.rstrip("/").split("/")[-1] not in ("#", "health", "")]
    skipped_invalid = len(ARTICLE_URLS) - len(valid_urls)

    total   = len(valid_urls)
    saved   = 0
    skipped = 0
    blocked = 0
    already = 0

    print(f"\n{'='*60}")
    print(f"  Vishvas News Health Fact-Check Scraper")
    print(f"{'='*60}")
    if skipped_invalid:
        print(f"  ⛔ Auto-filtered {skipped_invalid} non-article URL(s)")
    print(f"  Articles to scrape : {total}")
    print(f"  Output             : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    session = requests.Session()
    # Warm up
    try:
        session.get("https://www.vishvasnews.com/", headers=HEADERS, timeout=10)
        time.sleep(random.uniform(1.0, 2.0))
    except Exception:
        pass

    for i, url in enumerate(valid_urls, start=1):
        slug = url_to_slug(url)

        if already_saved(slug):
            print(f"  [{i:03d}/{total}] ⏭️  already saved: {slug[:50]}")
            already += 1
            continue

        print(f"  [{i:03d}/{total}] {slug[:55]:<55}", end="  ", flush=True)

        title, text = fetch_and_extract(url, session)

        if text == "BLOCKED":
            print("🚫  blocked")
            blocked += 1
            time.sleep(random.uniform(5.0, 8.0))
            continue

        if text.startswith("ERROR:") or not text:
            print(f"❌  failed")
            skipped += 1
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            continue

        word_count = len(text.split())
        if word_count < MIN_WORDS:
            print(f"⚠️  too short ({word_count} words)")
            skipped += 1
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            continue

        filepath = os.path.join(OUTPUT_DIR, f"vishvas__{slug}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"SOURCE: Vishvas News — Health Fact Checks\n")
            f.write(f"TITLE: {title}\n")
            f.write(f"URL: {url}\n")
            f.write("─" * 60 + "\n\n")
            f.write(text)

        print(f"✅  ({word_count} words)")
        saved += 1
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    vishvas_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("vishvas__")]
    print(f"\n{'='*60}")
    print(f"  ✅  DONE")
    print(f"{'='*60}")
    print(f"  Saved this run   : {saved}")
    print(f"  Already existed  : {already}")
    print(f"  Skipped (thin)   : {skipped}")
    if blocked:
        print(f"  Blocked          : {blocked}  ← re-run to retry")
    print(f"  Total files in KB: {len(vishvas_files)}")
    print(f"  Output           : {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        import trafilatura, requests
        from bs4 import BeautifulSoup
    except ImportError as e:
        print(f"Missing: {e}\nRun: pip install requests beautifulsoup4 trafilatura")
        exit(1)
    main()