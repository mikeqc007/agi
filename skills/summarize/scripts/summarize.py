#!/usr/bin/env python3
"""Bulk summarize URLs from a text file (one URL per line)."""

import argparse
import urllib.request

def fetch(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.read().decode("utf-8", errors="replace")[:8000]
    except Exception as e:
        return f"[fetch error: {e}]"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="file with one URL per line")
    args = parser.parse_args()

    with open(args.input) as f:
        urls = [line.strip() for line in f if line.strip()]

    for url in urls:
        print(f"\n=== {url} ===")
        content = fetch(url)
        # Print first 500 chars as preview — agent will summarize the rest
        print(content[:500])

if __name__ == "__main__":
    main()
