#!/usr/bin/env python
"""
Build the small demo corpus that ships with the live web demo.

Pulls the first N articles of the English Wikipedia dump (20231101.en — the same
source used for the full 100K-document benchmark) through the public Hugging Face
datasets-server API, truncates each article, and writes gzip-compressed JSONL to
demo_data/corpus.jsonl.gz.

Usage:
    python scripts/build_demo_corpus.py --docs 4000 --chars 3500
"""
import argparse
import gzip
import json
import os
import time

import requests

API = "https://datasets-server.huggingface.co/rows"
PARAMS = {"dataset": "wikimedia/wikipedia", "config": "20231101.en", "split": "train"}
PAGE = 100  # API maximum


def fetch(offset: int, retries: int = 5):
    for attempt in range(retries):
        r = requests.get(API, params={**PARAMS, "offset": offset, "length": PAGE}, timeout=60)
        if r.status_code == 200:
            return [row["row"] for row in r.json()["rows"]]
        time.sleep(2 ** attempt)
    r.raise_for_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=4000)
    ap.add_argument("--chars", type=int, default=3500, help="truncate article text")
    ap.add_argument("--out", default="demo_data/corpus.jsonl.gz")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    written = 0
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        for offset in range(0, args.docs, PAGE):
            for row in fetch(offset):
                text = row["text"][: args.chars]
                if len(text) < 300:  # skip stubs / disambiguation pages
                    continue
                f.write(json.dumps({
                    "doc_id": f"wiki_{row['id']}",
                    "title": row["title"],
                    "url": row["url"],
                    "content": text,
                }, ensure_ascii=False) + "\n")
                written += 1
            print(f"  fetched {offset + PAGE} rows, kept {written}")
    print(f"Wrote {written} documents to {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
