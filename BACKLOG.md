# Backlog — future versions

Deferred on purpose; nothing here blocks v2 of the paper.

## Cut from paper draft v2 (per author feedback, 2026-07-23)

- **Cloud / rented-hardware guidance** ("No 16 GB GPU? No Mac?"): rented CUDA
  boxes (RunPod, Vast.ai, Colab paid tiers) for readers without hardware;
  cloud Macs (AWS EC2 Mac) exist but are CI-priced with 24-hour minimums.
  Needs validation on at least one rented environment before publishing —
  we promise measured claims, so we should rent one and measure.

## Repo features the paper already references as "on the backlog"

- `--bits` flag for `scripts/cpu_demo.py` (so Ch 2's Try-it doesn't require
  editing source), plus an MSE-grid-without-rotation ablation arm — the arm
  also lets Graphic B credit rotation honestly.
- `scripts/context_ceiling.py`: automate the Ch 6 engine-init bisection.
- `--depths` convenience: consider adding a depth flag to demo.py to match
  benchmark.py.

## Published-docs corrections surfaced by the outline fact-check (2026-07-21)

- README.md: "3.4x more parallel 40K requests" conflates absolute
  concurrency (3.43) with the improvement ratio (~3.2x).
- docs/apple-silicon.md: "Replaying every failing prompt… 52/52" overstates
  scope — it was the four originally-failing prompts, 13 replays each.
- LEARN.md: the "8x" figure is from Google's blog on the ICLR presentation,
  not the arXiv paper — fix attribution.

## Paper / experiment extensions (later rounds)

- Produce final graphics A–L from the validated prompts once language is
  approved; regenerate Graphic C from `results/*.json`.
- Measure the 60-trial CUDA benchmark wall-clock time (Ch 4 currently says
  "tens of minutes per condition" as an estimate).
- Multi-needle and 64K+ context benchmark tiers (extends the claims the
  needle test can support).
- `turboquant_k8v4` and `turboquant_3bit_nc` rungs: run locally so the Ch 7
  ladder has no "not run here" rows.
- Optional: TheTom fork exploration for real TurboQuant types on Metal
  (maintainer note 9).
