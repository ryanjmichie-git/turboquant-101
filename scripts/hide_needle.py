#!/usr/bin/env python3
"""Hide your own needle: one secret, one long document, compression off then on.

Hides one sentence (the "needle") in a long document, then asks Qwen3-4B
to find it twice: once with the normal KV cache, once with TurboQuant
(turboquant_4bit_nc). Each run starts the model fresh and reports vLLM's
own "GPU KV cache size" startup line, so the room each cache gives you
prints next to the answer it gave.

With no --file, it downloads Pride and Prejudice from Project Gutenberg
(public domain) the first time, and uses its first 25,000 tokens.

  python scripts/hide_needle.py
  python scripts/hide_needle.py \\
      --needle "Mr. Bennet keeps a spare key under the blue flowerpot." \\
      --question "Where does Mr. Bennet keep his spare key?"
  python scripts/hide_needle.py --file /mnt/c/Users/YOU/Documents/notes.txt \\
      --needle "..." --question "..."
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BOOK_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
BOOK = ROOT / ".venv" / "tq101-books" / "pride-and-prejudice.txt"
COMPRESSED = "turboquant_4bit_nc"
DOC_TOKENS = 25_000
MAX_MODEL_LEN = 32_768
GPU_MEM_UTIL = 0.85  # verify.py's budget, so the KV numbers match Chapter 4
KV_LINE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")


def fetch_book() -> pathlib.Path:
    """Download Pride and Prejudice once, minus Gutenberg's header/footer."""
    if not BOOK.exists():
        print(
            "Downloading Pride and Prejudice from Project Gutenberg "
            "(public domain, under 1 MB)...",
            flush=True,
        )
        raw = (
            urllib.request.urlopen(BOOK_URL, timeout=120)
            .read()
            .decode("utf-8", errors="replace")
        )
        start = raw.find("*** START")
        start = raw.find("\n", start) + 1 if start != -1 else 0
        end = raw.find("*** END")
        end = end if end != -1 else len(raw)
        BOOK.parent.mkdir(parents=True, exist_ok=True)
        BOOK.write_text(raw[start:end].strip().replace("\r\n", "\n"), encoding="utf-8")
    return BOOK


def ask(
    cache_dtype: str | None, doc: pathlib.Path, needle: str, question: str, depth: int
) -> None:
    """Child process: start the engine, hide the needle, ask once."""
    from niah import VLLMBackend

    backend = VLLMBackend(
        cache_dtype=cache_dtype,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enforce_eager=True,
    )
    try:
        ids = backend.tokenizer.encode(
            doc.read_text(encoding="utf-8", errors="replace")
        )
        text = backend.tokenizer.decode(ids[:DOC_TOKENS])
        # Hide it between two lines near the requested depth, not mid-word.
        at = len(text) * depth // 100
        cut = text.rfind("\n", 0, at)
        cut = cut if cut > 0 else at
        prompt = f"{text[:cut]}\n{needle}\n{text[cut:]}\n\n{question}"
        res = backend.generate(prompt)
    finally:
        backend.close()
    print(
        "RESULT "
        + json.dumps(
            {
                "answer": res.text.strip(),
                "doc_tokens": min(len(ids), DOC_TOKENS),
                "file_tokens": len(ids),
            }
        ),
        flush=True,
    )


def run(
    cache_dtype: str | None, args, doc: pathlib.Path
) -> tuple[dict | None, int | None]:
    """Parent: one fresh child per condition; returns (result, kv_tokens)."""
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--_child",
        cache_dtype or "",
        "--file",
        str(doc),
        "--needle",
        args.needle,
        "--question",
        args.question,
        "--depth",
        str(args.depth),
    ]
    # The "GPU KV cache size" line is logged at INFO.
    env = dict(
        os.environ, VLLM_LOGGING_LEVEL=os.environ.get("VLLM_LOGGING_LEVEL", "INFO")
    )
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log = proc.stdout + "\n" + proc.stderr
    print(f" done ({time.perf_counter() - t0:.0f} s)", flush=True)
    kv = KV_LINE.search(log)
    kv_tokens = int(kv.group(1).replace(",", "")) if kv else None
    hit = re.search(r"^RESULT (\{.*\})$", log, re.M)
    if proc.returncode != 0 or not hit:
        print("  FAILED. Last lines of the engine log:")
        print("\n".join("    " + line for line in log.strip().splitlines()[-15:]))
        return None, kv_tokens
    return json.loads(hit.group(1)), kv_tokens


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--file",
        type=pathlib.Path,
        help="a plain-text (.txt) document to hide the needle in "
        "(default: Pride and Prejudice)",
    )
    ap.add_argument(
        "--needle",
        default="The secret password is PURPLE-OTTER-42.",
        help="the sentence to hide",
    )
    ap.add_argument(
        "--question",
        default="What is the secret password? Reply with the password only.",
        help="what to ask after the document",
    )
    ap.add_argument(
        "--depth",
        type=int,
        default=50,
        help="where to hide the needle, in percent of the document "
        "(default: 50, the middle)",
    )
    ap.add_argument("--_child", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._child is not None:
        ask(args._child or None, args.file, args.needle, args.question, args.depth)
        return

    doc = args.file.expanduser().resolve() if args.file else fetch_book()
    if not doc.is_file():
        ap.error(f"no such file: {doc}")
    name = "Pride and Prejudice" if doc == BOOK else doc.name
    print(f"\nDocument: {name}")
    print(f"Needle:   {args.needle!r}, hidden {args.depth}% of the way in")
    print(f"Question: {args.question!r}")

    runs = []
    for i, (label, dtype) in enumerate(
        [
            ("compression OFF (normal cache)", None),
            (f"compression ON ({COMPRESSED})", COMPRESSED),
        ],
        1,
    ):
        print(
            f"\nRun {i}/2: {label}. Loading the model and reading the document...",
            end="",
            flush=True,
        )
        res, kv = run(dtype, args, doc)
        runs.append((res, kv))
        if kv:
            more = f"  ({kv / runs[0][1]:.2f}x run 1)" if i == 2 and runs[0][1] else ""
            print(f"  Room in the KV cache: {kv:,} tokens{more}")
        if res:
            if i == 1:
                of = (
                    f" of {res['file_tokens']:,}"
                    if res["file_tokens"] > res["doc_tokens"]
                    else ""
                )
                print(f"  Model read: {res['doc_tokens']:,}{of} tokens of the document")
            print(f"  Answer: {res['answer']}")

    (off, kv_off), (on, kv_on) = runs
    print()
    if not (off and on):
        print(
            "One run failed, so there is nothing to compare. See the log lines above."
        )
    elif off["answer"].casefold() == on["answer"].casefold():
        ratio = (
            f", with {kv_on / kv_off:.2f}x more room in the cache"
            if kv_off and kv_on
            else ""
        )
        print(f"Same answer both times{ratio}.")
        print(
            "Check it against the needle you hid: same answer is only proof if it's right."
        )
    else:
        print("The wording differs between the two runs. Read both answers: does each")
        print("one contain the fact you hid? Different words can carry the same fact.")


if __name__ == "__main__":
    main()
