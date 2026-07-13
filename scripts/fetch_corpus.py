#!/usr/bin/env python3
"""(Re)build data/corpus.txt from a public-domain source.

The repo ships with the corpus already bundled, so you only need this if
you deleted it or want a different text. Any long, neutral English prose
works -- the filler just needs to be realistic enough that the model can't
trivially spot the needle by style alone.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import urllib.request

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "corpus.txt"
# Darwin, "The Voyage of the Beagle" (Project Gutenberg ebook #944).
DEFAULT_URL = "https://www.gutenberg.org/cache/epub/944/pg944.txt"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--max-chars", type=int, default=1_200_000)
    args = ap.parse_args()

    print(f"Fetching {args.url} ...")
    raw = urllib.request.urlopen(args.url, timeout=120).read().decode(
        "utf-8", errors="replace"
    )

    # Strip Project Gutenberg header/footer boilerplate. Their trademark
    # policy asks for this when redistributing the plain text; the work
    # itself is public domain.
    start = raw.find("*** START")
    start = raw.find("\n", start) + 1 if start != -1 else 0
    end = raw.find("*** END")
    end = end if end != -1 else len(raw)
    body = re.sub(r"\r\n", "\n", raw[start:end].strip())[: args.max_chars]
    cut = body.rfind(". ")
    body = body[: cut + 1] if cut > 0 else body

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(body)
    print(f"Wrote {OUT} ({len(body):,} chars, ~{len(body)//4:,} tokens est.)")


if __name__ == "__main__":
    main()
