"""Reproducible synthetic demonstration (no competition data).

Generates a toy panel of companies reporting once per period. The data-generating process
has the economic structure the model is built around:

  * earnings surprises persist (this period's surprise partly repeats last period's),
  * the market prices the EXPECTED part of a surprise, so given today's surprise, a
    high prior surprise means a smaller reaction (a repeat beat is partly priced in),
  * disclosure text carries news beyond the headline surprise,
  * reaction size depends on firm volatility and release timing.

Walk-forward, point-in-time evaluation: the quant model is refit on all earlier periods,
the prior-surprise coefficient is fit on OUT-OF-SAMPLE base predictions, prior surprises
are looked up with the point-in-time table, and the last period is held out.

    python demo.py            # prints the comparison table (about 10-20 seconds)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from earnings_reaction import (
    PriorSurpriseAdjustment,
    PriorSurpriseTable,
    QuantModel,
    delta_r2,
    percentile_rank,
    residualize,
)

POS = ["revenue grew strongly", "full year guidance raised", "record backlog", "margins improved",
       "demand accelerated"]
NEG = ["demand weakened", "guidance lowered", "margin headwinds", "revenue declined", "costs exceeded plan"]
NEUTRAL = ["the dividend was maintained", "headcount was flat", "capital spending was unchanged",
           "the board approved the buyback", "inventory was in line"]
START = datetime(2025, 1, 1, tzinfo=timezone.utc)


def simulate(n_firms: int = 500, n_periods: int = 5, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0.5, 1.5, n_firms)
    prev_s = np.zeros(n_firms)
    events = []
    for q in range(n_periods):
        s = 0.6 * prev_s + rng.standard_normal(n_firms)          # persistent surprises
        news = rng.standard_normal(n_firms)                        # information in the text
        bmo = rng.random(n_firms) < 0.55
        day = rng.integers(5, 60, n_firms)
        sigma = vol * np.where(bmo, 1.3, 1.0)
        car = sigma * (0.5 * s - 0.15 * prev_s + 0.45 * news + rng.standard_normal(n_firms))
        y, s_pct = percentile_rank(car), percentile_rank(s)
        for i in range(n_firms):
            k = rng.integers(6, 10)
            p_pos = 1 / (1 + np.exp(-(news[i] + 0.3 * s[i])))
            sentences = [rng.choice(POS) if rng.random() < p_pos * 0.7 else
                         rng.choice(NEG) if rng.random() < (1 - p_pos) * 0.7 else rng.choice(NEUTRAL)
                         for _ in range(k)]
            sentences.append(f"sales changed {rng.normal(5 * news[i], 4):.1f}%")
            released = START + timedelta(days=91 * q + int(day[i]), hours=12 if bmo[i] else 21)
            new_listing = q == n_periods - 1 and i % 10 == 0  # 10% have no reporting history
            events.append({"ticker": f"{'N' if new_listing else 'T'}{i:03d}", "period": f"P{q + 1}",
                           "released_at": released,
                           "knowledge_cutoff": released - timedelta(hours=16),
                           "text": ". ".join(sentences), "timing": [float(bmo[i]), float(day[i])],
                           "y": float(y[i]), "surprise_pct": float(s_pct[i])})
        prev_s = s
    return events


def period_final_at(n_periods: int) -> dict[str, datetime]:
    # ranks are final once the whole period has reported (+ a publication lag)
    return {f"P{q + 1}": START + timedelta(days=91 * q + 63) for q in range(n_periods)}


def columns(ev: list[dict]):
    return ([e["text"] for e in ev], np.array([e["timing"] for e in ev]), np.array([e["y"] for e in ev]),
            np.array([e["surprise_pct"] for e in ev]))


def run(n_firms: int = 500, n_periods: int = 5, seed: int = 7, missing: float = 0.15, verbose: bool = True) -> dict:
    events = simulate(n_firms, n_periods, seed)
    table = PriorSurpriseTable(period_final_at(n_periods))
    rng = np.random.default_rng(seed + 1)
    for e in events:  # some surprises are never published (e.g. no consensus): no table entry
        if rng.random() >= missing:
            table.add(e["ticker"], e["released_at"], e["surprise_pct"], e["period"])
    by = {p: [e for e in events if e["period"] == p] for p in sorted({e["period"] for e in events})}
    periods = list(by)
    test_p = periods[-1]

    # 1. Out-of-sample base predictions for the periods used to fit the adjustment.
    oof_base, oof_prev, oof_res = [], [], []
    for k in range(1, len(periods) - 1):
        train = [e for p in periods[:k] for e in by[p]]
        model = QuantModel().fit(*columns(train))
        ev = by[periods[k]]
        texts, timing, y, s = columns(ev)
        oof_base.append(model.predict(texts, timing))
        oof_prev.append([np.nan if (v := table.lookup(e["ticker"], e["released_at"], e["knowledge_cutoff"])) is None
                         else v for e in ev])
        oof_res.append(residualize(y, s))
    adj = PriorSurpriseAdjustment().fit(np.concatenate(oof_base), np.concatenate(oof_prev), np.concatenate(oof_res))

    # 2. Final quant fit on every earlier period; predict the held-out period.
    model = QuantModel().fit(*columns([e for p in periods[:-1] for e in by[p]]))
    test = by[test_p]
    texts, timing, y, s = columns(test)
    d, m = model.components(texts, timing)
    base = model.predict(texts, timing)
    # Live webhooks may lack the cutoff: use the conservative bound (event_datetime - margin).
    prev = [table.lookup(e["ticker"], e["released_at"]) for e in test]
    final = np.array([adj.apply(b, p) for b, p in zip(base, prev, strict=True)])

    fallback_exact = all(f == b for f, b, p in zip(final, base, prev, strict=True) if p is None)
    res = {"held_out_period": test_p, "events": len(test), "fallback_exact": fallback_exact,
           "prior_surprise_coverage": float(np.mean([p is not None for p in prev])),
           "fitted_gamma_sign": "negative" if adj.gamma < 0 else "non-negative",
           "delta_r2": {"text direction only": delta_r2(y, s, d),
                        "direction x magnitude (quant base)": delta_r2(y, s, base),
                        "+ prior-quarter surprise adjustment": delta_r2(y, s, final)}}
    if verbose:
        print(f"Synthetic held-out period {test_p}: {len(test)} events, "
              f"prior surprise available for {res['prior_surprise_coverage']:.0%} "
              f"(others fall back exactly: {fallback_exact}), gamma {res['fitted_gamma_sign']}")
        print(f"{'model':<40s} {'delta R^2':>10s}")
        for k, v in res["delta_r2"].items():
            print(f"{k:<40s} {v:>10.4f}")
        print("(synthetic data: illustrates the mechanism, not competition performance)")
    return res


if __name__ == "__main__":
    run()
