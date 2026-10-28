"""Can the label be predicted from the claim text alone, without evidence?

If a style-only or bag-of-words classifier beats the majority class by a wide margin, the
dataset rewards surface cues and pipeline accuracy on it is confounded. Report these
baselines next to any pipeline score.
"""

import argparse
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline


def style_features(text: str) -> list:
    return [
        len(text.split()), len(text), bool(re.search(r"\d", text)),
        bool(re.search(r"\b(not|no|never|without)\b", text, re.I)),
        bool(re.search(r"\b(may|might|can|could|some|often|likely|suggest)\b", text, re.I)),
        text.count(","), text.strip().endswith("."), bool(re.match(r"^[a-z]", text)),
        sum(c.isupper() for c in text) / max(1, len(text)),
        bool(re.search(r"\b(cure|cures|kills|always|never|all|100%|miracle)\b", text, re.I)),
    ]


def check(texts: pd.Series, labels: pd.Series) -> dict:
    folds = min(5, int(labels.value_counts().min()))
    cv = StratifiedKFold(folds, shuffle=True, random_state=0)
    tfidf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=2000))
    style = np.array([style_features(t) for t in texts], dtype=float)
    return {
        "n": len(texts),
        "majority": float(labels.value_counts(normalize=True).max()),
        "tfidf_no_evidence": float(cross_val_score(tfidf, texts, labels, cv=cv).mean()),
        "style_only": float(cross_val_score(LogisticRegression(max_iter=2000), style, labels, cv=cv).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--text", default="claim")
    ap.add_argument("--label", default="label")
    ap.add_argument("--keep-labels", default=None, help="comma list of label values to keep")
    ap.add_argument("--group", default=None, help="also report per value of this column")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.keep_labels:
        df = df[df[args.label].astype(str).isin(args.keep_labels.split(","))]
    groups = [("ALL", df)] + (list(df.groupby(args.group)) if args.group else [])
    print(f"{'subset':22} {'n':>6} {'majority':>9} {'tfidf':>7} {'style':>7}")
    for name, g in groups:
        if g[args.label].nunique() < 2 or len(g) < 40:
            continue
        r = check(g[args.text].astype(str), g[args.label].astype(str))
        flag = "  <- style leak" if r["style_only"] - r["majority"] > 0.05 else ""
        print(f"{str(name):22} {r['n']:>6} {r['majority']:>9.3f} {r['tfidf_no_evidence']:>7.3f} {r['style_only']:>7.3f}{flag}")


if __name__ == "__main__":
    main()
