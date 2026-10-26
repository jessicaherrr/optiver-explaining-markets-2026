# Optiver Explaining Markets 2026 — Quantitative Earnings-Reaction Forecasting

A statistical model for the [Explaining Markets](https://explainingmarkets.ai) 2026
contest. For each earnings release, it predicts where the company's next-day
abnormal return will rank among all reactions in the quarter.

This repository is a compact, self-contained **reference implementation** of the
architecture, with a synthetic-data demonstration and tests. The live contest system
is not included (see [Production vs. this repository](#production-vs-this-repository)).

## 1. Overview

The contest scores each submission by how much of the realized reaction ranking it
explains **beyond the earnings surprise**:

ΔR² = R²(rank of reaction ~ prediction + surprise rank) − R²(rank of reaction ~ surprise rank)

A good model must therefore find information the headline beat or miss does not
already contain. Two sources do most of the work:

- **The disclosure text.** Guidance, margins, demand and tone, read by a regularized
  linear text model.
- **Market expectations.** A company that beat strongly last quarter is expected to
  beat again, so part of a repeat beat is already priced in.

## 2. Model architecture

```
fact summary ──► TF-IDF n-grams + financial lexicon ──► Ridge ─────────► direction d
release timing, disclosure length / text signal ─────► LightGBM ──────► magnitude m ≈ E|y − ½|

base percentile   p₀ = ½ + k · z(d · m)
prior surprise    s₋₁ = the company's previous earnings-surprise percentile (point-in-time lookup)
final percentile  p  = clip(p₀ + γ · z(s₋₁), 0.01, 0.99)       if s₋₁ is available
                  p  = p₀                                        otherwise (exact fallback)
```

- **Direction.** Ridge regression on TF-IDF features and lexicon counts. It is
  trained on the *surprise-residualized* reaction rank, the only part the metric
  rewards.
- **Magnitude.** A LightGBM regressor predicts how large the reaction will be, so
  confident directional calls on volatile names count for more.
- **Prior-quarter surprise adjustment.** A single linear term with a negative
  coefficient (a repeat beat is partly priced in). It is fit on out-of-sample base
  predictions and held fixed.
- **No language model, no network, no external data** at prediction time.
  Inference is deterministic and takes milliseconds.

## 3. Point-in-time methodology and evaluation

- **Information set.** Only the event's official fact summary, its release time,
  and earlier quarters of the competition's own archive.
- **Point-in-time lookup.** A prior surprise is used only if:
  - it was released before the event's knowledge cutoff, or, when the cutoff is not
    supplied, before a conservative bound earlier than any true cutoff; and
  - the percentile ranks of its quarter were already final at that time.

  Anything else falls back to the base prediction, unchanged.
- **Validation.** An expanding walk-forward by quarter (fit on all earlier quarters,
  score the next) with the contest metric. Uncertainty comes from paired bootstrap
  resampling of event *days*, because reactions cluster by day.
- **Design rules.**
  - A small number of pre-specified candidates, with a fixed promotion rule.
  - No tuning on the most recent hold-out period.
  - Every component must improve the metric on separated data, not just in sample.

## 4. Reproducible synthetic demonstration

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
python demo.py
pytest -q
```

`demo.py` simulates companies reporting over five periods. The data are built so that:
- surprises persist from one period to the next;
- the market prices the expected part of a surprise;
- the text carries extra news;
- reaction size depends on volatility and timing.

It then runs the full walk-forward protocol on the last period. Expected output:

```
Synthetic held-out period P5: 500 events, prior surprise available for 90% (others fall back exactly: True), gamma negative
model                                     delta R^2
text direction only                          0.0292
direction x magnitude (quant base)           0.0318
+ prior-quarter surprise adjustment          0.0382
```

These numbers come from synthetic data. They show the mechanism, not contest
performance.

## Production vs. this repository

| | Production contest model | This repository |
|---|---|---|
| Architecture | As in §2 | Same structure |
| Lexicon, n-gram settings, regularization, tree settings | Tuned on the contest archive | Small illustrative defaults |
| Prior-surprise coefficient | Fit once on archive quarters, then frozen | Fit on the synthetic data you run |
| Data | Licensed contest archive and live event payloads | Synthetic only |
| Metric | Official scorer | Short re-implementation of ΔR² (mean imputation for missing predictions) |
| Serving | Webhook service with logging, deadline handling and fallbacks | Not included |

Trained weights, production coefficients and contest data are deliberately not
published while the contest is running.

## 5. LLM research extension (research only)

The submitted model is the quantitative model above. It uses no language model.
Two LLM research tracks run alongside it and never affect a submitted prediction:

- **Gemini Independent shadow.** After each official prediction has been submitted,
  a separate service asks Gemini to read the same event materials on its own and
  records its percentile next to the quantitative prediction. These records are
  collected prospectively, before outcomes exist, for a pre-registered comparison
  of the quantitative model with fixed, normalized Quant–Gemini blends. If a Gemini
  call fails, nothing happens to the submission.
- **Expectation-Aware Quant–LLM research.** A separate project studies whether an
  LLM can turn disclosures into structured, evidence-cited financial features, such
  as guidance revisions and results relative to expectations, for a regularized
  quantitative model to weight.

Neither track is part of the deployed model, neither is included in this
repository, and neither has yet shown superior performance. Any future use would
first have to pass prospective, out-of-sample evaluation.

## 6. Limitations and disclaimer

- **The evidence is still developing.** The prior-surprise effect is small, it is
  estimated on a few quarters of history, and it remains uncertain. Text features
  read a ten-sentence summary, not full transcripts.
- **Synthetic results** illustrate the method and are not forecasts of contest
  results.
- **Independent participation.** This is an independent contest entry and personal
  research project. It is not affiliated with, sponsored by or endorsed by Optiver,
  the Explaining Markets organizers or their partners. All trademarks belong to their
  owners. No contest data, organizer code or confidential information is included.

## License

MIT, © 2026 Jessica He. See [LICENSE](LICENSE).
