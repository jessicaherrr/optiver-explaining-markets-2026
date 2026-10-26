from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import demo
from earnings_reaction import (
    PriorSurpriseAdjustment,
    PriorSurpriseTable,
    QuantModel,
    delta_r2,
    percentile_rank,
    residualize,
)

UTC = timezone.utc
T0 = datetime(2026, 1, 1, tzinfo=UTC)


# -- metric -------------------------------------------------------------------


def test_percentile_rank_handles_ties_and_missing():
    r = percentile_rank(np.array([3.0, 1.0, 3.0, np.nan]))
    assert r[1] == pytest.approx(1 / 3) and r[0] == r[2] == pytest.approx(2.5 / 3) and np.isnan(r[3])


def test_delta_r2_rewards_information_beyond_the_surprise():
    rng = np.random.default_rng(0)
    s, a = rng.standard_normal(2000), rng.standard_normal(2000)
    y = percentile_rank(s + a + rng.standard_normal(2000))
    sp = percentile_rank(s)
    assert delta_r2(y, sp, a) > 0.1
    assert delta_r2(y, sp, sp) == pytest.approx(0.0, abs=1e-9)  # the surprise itself adds nothing
    assert delta_r2(y, sp, -a) == pytest.approx(delta_r2(y, sp, a))  # sign-invariant regression metric
    assert delta_r2(y, sp, rng.standard_normal(2000)) < 0.005


def test_delta_r2_mean_imputes_missing_predictions():
    rng = np.random.default_rng(1)
    y, s, p = rng.random(300), rng.random(300), rng.random(300)
    p_missing = p.copy()
    p_missing[:50] = np.nan
    filled = p_missing.copy()
    filled[:50] = np.nanmean(p_missing)
    assert delta_r2(y, s, p_missing) == pytest.approx(delta_r2(y, s, filled))


def test_residualize_removes_the_surprise_projection():
    rng = np.random.default_rng(2)
    s = rng.random(500)
    y = 0.7 * s + 0.1 * rng.standard_normal(500)
    assert abs(np.corrcoef(residualize(y, s), s)[0, 1]) < 1e-8


# -- point-in-time lookup -----------------------------------------------------


def table():
    t = PriorSurpriseTable({"P1": T0 + timedelta(days=70), "P2": T0 + timedelta(days=160)})
    t.add("A", T0 + timedelta(days=20), 0.2, "P1")
    t.add("A", T0 + timedelta(days=110), 0.9, "P2")
    return t


def test_lookup_requires_release_and_final_ranks_before_the_bound():
    t = table()
    assert t.lookup("A", None, T0 + timedelta(days=200)) == 0.9
    # P2 entry released but its period's ranks are not final yet -> use the last final period
    assert t.lookup("A", None, T0 + timedelta(days=130)) == 0.2
    assert t.lookup("A", None, T0 + timedelta(days=20)) is None  # same instant: not yet released
    assert t.lookup("UNKNOWN", None, T0 + timedelta(days=200)) is None


def test_lookup_prefers_the_cutoff_and_otherwise_uses_a_conservative_bound():
    t = table()
    release = T0 + timedelta(days=163)
    assert t.bound(release, None) == release - t.fallback_margin
    assert t.bound(release, T0) == T0
    assert t.lookup("A", release) == 0.2  # release - 120 h < P2 final time
    assert t.lookup("A", None, None) is None


# -- adjustment ---------------------------------------------------------------


def test_missing_prior_returns_the_base_prediction_exactly():
    adj = PriorSurpriseAdjustment(gamma=-0.05)
    assert adj.apply(0.5371, None) == 0.5371
    assert adj.apply(0.5371, float("nan")) == 0.5371
    assert PriorSurpriseAdjustment(gamma=-10.0).apply(0.5, 1.0) == 0.01  # clipped


def test_adjustment_learns_a_negative_coefficient_when_repeat_beats_are_priced_in():
    rng = np.random.default_rng(3)
    base, prev = rng.normal(0.5, 0.05, 3000), rng.random(3000)
    y_res = 2 * (base - 0.5) - 0.2 * (prev - 0.5) + 0.2 * rng.standard_normal(3000)
    assert PriorSurpriseAdjustment().fit(base, prev, y_res).gamma < 0


# -- quant model and demo -------------------------------------------------------


def test_quant_model_gives_one_valid_deterministic_prediction_per_event():
    ev = demo.simulate(n_firms=120, n_periods=2, seed=0)
    texts, timing, y, s = demo.columns(ev)
    p1 = QuantModel().fit(texts, timing, y, s).predict(texts, timing)
    p2 = QuantModel().fit(texts, timing, y, s).predict(texts, timing)
    assert p1.shape == (len(ev),) and np.all((p1 >= 0) & (p1 <= 1))
    np.testing.assert_array_equal(p1, p2)


def test_demo_runs_end_to_end():
    res = demo.run(n_firms=150, n_periods=4, seed=1, verbose=False)
    assert res["fallback_exact"] and 0 < res["prior_surprise_coverage"] < 1
    assert set(res["delta_r2"]) == {"text direction only", "direction x magnitude (quant base)",
                                    "+ prior-quarter surprise adjustment"}
