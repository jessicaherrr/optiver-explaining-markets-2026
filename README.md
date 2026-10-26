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

Two quantitative branches, one for **direction** and one for **magnitude**, form a
baseline prediction. A **prior-quarter surprise adjustment** is then applied. The
equations below describe the public reference implementation in
[`earnings_reaction.py`](earnings_reaction.py). The production model has the same
structure, with its own features, settings and fitted coefficients, which are not
published.

| Symbol | Meaning |
|---|---|
| $`y_i`$ | Percentile rank of event $`i`$'s one-day abnormal return within the period |
| $`s_i`$ | Percentile rank of event $`i`$'s earnings surprise within the period |
| $`r_i`$ | Surprise-residualized reaction, the Ridge target |
| $`\mathbf{x}_i`$ | TF-IDF n-grams and standardized financial-lexicon counts |
| $`\mathbf{z}_i`$ | Release-timing and text-derived features |
| $`d_i,\ m_i`$ | Directional signal and expected reaction magnitude |
| $`p_i^{0},\ p_i`$ | Baseline and final predicted percentile |
| $`s_i^{\mathrm{prev}}`$ | The company's prior earnings-surprise percentile, if eligible |
| $`\mu,\ \sigma`$ | Training-period mean and standard deviation of the subscripted quantity |
| $`k,\ \gamma`$ | Calibration slope and prior-surprise coefficient |
| $`B_i`$ | Point-in-time information bound for event $`i`$ |

### Stage 1: directional signal

The metric rewards only the part of the reaction that the surprise does not explain.
The Ridge target therefore removes the surprise's linear projection:

```math
r_i = y_i - \bar{y} - b\,(s_i - \bar{s}),
\qquad
b = \frac{\mathrm{Cov}(s, y)}{\mathrm{Var}(s)}
```

A Ridge regression on TF-IDF n-grams and lexicon counts then gives the directional
signal:

```math
d_i = \hat{\beta}_0 + \hat{\boldsymbol\beta}^{\top}\mathbf{x}_i,
\qquad
(\hat{\beta}_0, \hat{\boldsymbol\beta}) = \arg\min_{\beta_0,\,\boldsymbol\beta}
\sum_i \left(r_i - \beta_0 - \boldsymbol\beta^{\top}\mathbf{x}_i\right)^2
+ \lambda \lVert \boldsymbol\beta \rVert_2^2
```

### Stage 2: reaction magnitude

A LightGBM regressor $`\hat{g}`$, trained on release-timing and text-derived features,
estimates how far the reaction will land from the median. Volatile names get a
larger magnitude, so confident directional calls on them count for more:

```math
m_i = \max\left\{\hat{g}(\mathbf{z}_i),\ 10^{-3}\right\},
\qquad
\hat{g}(\mathbf{z}) \approx \mathbb{E}\left[\,\left\lvert y - \tfrac{1}{2} \right\rvert \,\middle|\, \mathbf{z}\,\right]
```

### Stage 3: quantitative baseline

Direction and magnitude are multiplied, standardized with training-period statistics,
and mapped affinely onto the percentile scale. The metric is a linear regression, so
the slope $`k`$ only sets the scale; clipping keeps every prediction a valid
percentile:

```math
p_i^{0} = \mathrm{clip}\left(\frac{1}{2} + k\,\frac{d_i m_i - \mu_{dm}}{\sigma_{dm}},\ 0.01,\ 0.99\right)
```

### Stage 4: prior-quarter surprise adjustment

A company that beat strongly last quarter is expected to beat again, so a repeat
beat is partly priced in. The adjustment is a single linear term with a **negative**
coefficient. It is fit on *out-of-sample* baseline predictions, then held fixed.
Without an eligible prior surprise, the baseline is returned **exactly**:

```math
p_i =
\begin{cases}
\mathrm{clip}\left(p_i^{0} + \gamma\,\dfrac{s_i^{\mathrm{prev}} - \mu_{s}}{\sigma_{s}},\ 0.01,\ 0.99\right)
& \text{if } s_i^{\mathrm{prev}} \text{ is eligible,} \\[8pt]
p_i^{0} & \text{otherwise.}
\end{cases}
```

**Eligibility.** The prior surprise comes from the most recent earlier event
$`j`$ of the same company that satisfies two conditions:
- it was released before the bound;
- its quarter's percentile ranks were final before the bound (time $`\tau_{q(j)}`$),
  because a rank within an unfinished quarter would use later events.

```math
s_i^{\mathrm{prev}} = s_{j^\ast},
\qquad
j^\ast = \text{latest } j \text{ with } t_j < B_i \text{ and } \tau_{q(j)} < B_i
```

```math
B_i =
\begin{cases}
\text{knowledge cutoff of event } i & \text{if it is supplied,} \\
t_i - \Delta & \text{otherwise.}
\end{cases}
```

Here $`t_i`$ is the release time. The margin $`\Delta`$ is chosen so the fallback
bound is never later than a true cutoff.

The prediction uses **no language model, no network and no external data**. It is
deterministic and takes milliseconds.

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
