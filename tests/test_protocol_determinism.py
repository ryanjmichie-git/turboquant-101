"""The A/B benchmark's core guarantee: baseline and compressed runs are
separate PROCESSES and must build byte-identical prompts. Python's builtin
hash() is salted per process, which silently broke this until the jitter
seed moved to zlib.crc32 -- these tests pin the fix."""

import os
import subprocess
import sys

from conftest import ROOT

CHILD = """
import hashlib
import sys
sys.path.insert(0, {root!r})
from niah.protocol import build_prompt, make_needles, make_trials
corpus = open({corpus!r}, encoding="utf-8").read()
digests = []
for trial in make_trials(make_needles(3), depths=[25, 50], contexts=[8192], reps=2):
    prompt = build_prompt(corpus, trial, lambda s: len(s) // 4)
    digests.append(hashlib.sha256(prompt.encode()).hexdigest())
print("|".join(digests))
"""


def _run_child(hash_seed: str) -> str:
    code = CHILD.format(
        root=str(ROOT), corpus=str(ROOT / "data" / "corpus.txt")
    )
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_prompts_identical_across_hash_seeds():
    results = {seed: _run_child(seed) for seed in ("0", "1", "424242")}
    assert len(set(results.values())) == 1, (
        "prompt construction depends on the process hash seed -- baseline "
        f"and compressed benchmark runs would diverge: {results}"
    )
