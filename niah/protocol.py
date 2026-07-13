"""Needle-in-a-haystack (NIAH) protocol.

The task: hide a short unique code ("the needle") inside a long stretch of
filler prose ("the haystack"), then ask the model to retrieve it. If the KV
cache compression corrupted the model's memory of the document, retrieval
fails. Exact-match retrieval of a unique string at 8-16K context is on the
EASY end of long-context evaluation -- treat a perfect score as "no damage
detectable by this task," not "no quality loss."

Reference protocol (from the known-good RTX 5080 run):
  5 needles x 3 depths (25/50/75%) x 2 contexts (8K/16K) x 2 reps = 60 trials
  per condition; baseline + compressed = 120 total. Greedy decoding, seed
  fixed, max_tokens=64.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Distinctive, easy-to-spot words for needle codes like GOLDFINCH-9241.
WORDS = [
    "GOLDFINCH", "LANTERN", "MERIDIAN", "OBSIDIAN", "PENUMBRA",
    "QUICKSILVER", "SAFFRON", "TAMARIND", "VERMILION", "ZEPHYR",
    "BASALT", "CARDAMOM", "DYNAMO", "EIDER", "FATHOM",
]

QUESTION = (
    "What is the secret code mentioned in the document? "
    "Reply with only the code."
)

PROMPT_TEMPLATE = (
    "Below is a long document. Read it carefully and answer the question "
    "at the end.\n\n<document>\n{document}\n</document>\n\n"
    "Question: {question}"
)


def make_needles(n: int, seed: int = 1234) -> list[str]:
    """Deterministically generate n unique WORD-NNNN codes."""
    rng = random.Random(seed)
    words = rng.sample(WORDS, n)
    return [f"{w}-{rng.randint(1000, 9999)}" for w in words]


def needle_sentence(code: str) -> str:
    return f"\nThe secret code is {code}. Remember the secret code.\n"


def _fit_to_tokens(text: str, target_tokens: int, count_tokens) -> str:
    """Trim text to approximately target_tokens, using the backend's own
    tokenizer for measurement. Character counts are only a starting guess
    (roughly 4 chars/token for English); we measure and re-trim a couple of
    times, which lands within a few percent of target. Exact token counts
    don't matter for NIAH -- consistency between conditions does.
    """
    chars = target_tokens * 4
    snippet = text[:chars]
    for _ in range(3):
        measured = count_tokens(snippet)
        if measured == 0:
            break
        ratio = target_tokens / measured
        if 0.97 <= ratio <= 1.03:
            break
        chars = max(200, int(len(snippet) * ratio))
        snippet = text[:chars]
    # End on a sentence boundary so the filler reads naturally.
    cut = snippet.rfind(". ")
    return snippet[: cut + 1] if cut > 200 else snippet


@dataclass
class Trial:
    needle: str
    depth_pct: int
    context_tokens: int
    rep: int


def build_prompt(
    corpus: str,
    trial: Trial,
    count_tokens,
    reserve_tokens: int = 220,
) -> str:
    """Assemble one NIAH prompt.

    The needle is inserted at depth_pct of the filler. Each rep shifts the
    corpus starting offset ("jitter") so repetitions aren't byte-identical
    -- with greedy decoding, identical prompts would give identical outputs
    and the extra reps would measure nothing.
    """
    filler_target = trial.context_tokens - reserve_tokens
    # Deterministic jitter per (needle, depth, rep) so runs are reproducible.
    rng = random.Random(hash((trial.needle, trial.depth_pct, trial.rep)) & 0xFFFF)
    max_offset = max(0, len(corpus) - filler_target * 5)
    offset = rng.randint(0, max_offset) if max_offset > 0 else 0
    filler = _fit_to_tokens(corpus[offset:], filler_target, count_tokens)

    split = int(len(filler) * trial.depth_pct / 100)
    # Snap to a sentence boundary near the target depth.
    snap = filler.rfind(". ", 0, split)
    split = snap + 2 if snap > 0 else split
    document = filler[:split] + needle_sentence(trial.needle) + filler[split:]
    return PROMPT_TEMPLATE.format(document=document, question=QUESTION)


def make_trials(
    needles: list[str],
    depths: list[int],
    contexts: list[int],
    reps: int,
) -> list[Trial]:
    return [
        Trial(needle=n, depth_pct=d, context_tokens=c, rep=r)
        for c in contexts
        for d in depths
        for n in needles
        for r in range(reps)
    ]


def score(response: str, needle: str) -> bool:
    """Exact-match: the full code appears anywhere in the response."""
    return needle.upper() in response.upper()
