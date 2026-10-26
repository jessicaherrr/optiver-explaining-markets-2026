"""Quantitative earnings-reaction forecasting: a self-contained reference implementation.

This module mirrors the ARCHITECTURE of the production model used for the Explaining
Markets 2026 contest. It is a simplified, independent re-implementation. The
hyperparameters, word lists and coefficients here are illustrative placeholders,
fitted only on whatever data you pass in. They are not the production configuration.

Pipeline for one (event, company):

    disclosure text --TF-IDF + lexicon--> Ridge --------------> direction  d
    release timing + text length  ------> LightGBM ----------> magnitude  m ~ E|y - 1/2|
    base percentile          = 1/2 + k * z(d * m)
    prior-quarter surprise   (point-in-time lookup, else fallback)
    final percentile         = clip(base + gamma * z(prior surprise))     gamma fit on data

Targets follow the contest: ``y`` is the within-period percentile rank of the one-day
abnormal return, and the score is the incremental R^2 over the period's percentile-ranked
earnings surprise (``delta_r2``).
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


def percentile_rank(x: np.ndarray) -> np.ndarray:
    """Average-rank percentiles in (0, 1]; NaN stays NaN."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    ok = ~np.isnan(x)
    out[ok] = rankdata(x[ok], method="average") / ok.sum()
    return out


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(y)), X])
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return 1.0 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))


def delta_r2(y: np.ndarray, surprise_pct: np.ndarray, pred: np.ndarray) -> float:
    """R^2(y ~ pred + surprise) - R^2(y ~ surprise) on rows with a surprise.

    Missing predictions are replaced by the mean prediction (as a contest-style
    imputation). A simplified re-implementation for illustration, not the official scorer.
    """
    y, s, p = (np.asarray(a, float) for a in (y, surprise_pct, pred))
    p = np.where(np.isnan(p), np.nanmean(p), p)
    m = ~np.isnan(s) & ~np.isnan(y)
    if m.sum() < 5 or np.std(p[m]) < 1e-12:
        return 0.0
    return float(_r2(y[m], np.column_stack([p[m], s[m]])) - _r2(y[m], s[m][:, None]))


def residualize(y: np.ndarray, surprise_pct: np.ndarray) -> np.ndarray:
    """The part of y the metric rewards: y minus its linear projection on the surprise."""
    y, s = np.asarray(y, float), np.asarray(surprise_pct, float)
    out = y - np.nanmean(y)
    m = ~np.isnan(s)
    if m.sum() > 2:
        b = np.cov(s[m], y[m], bias=True)[0, 1] / np.var(s[m])
        out[m] = y[m] - (y[m].mean() + b * (s[m] - s[m].mean()))
    return out


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

# Illustrative only: a handful of terms; the production lexicon is larger and not published.
POSITIVE = r"\b(beat|exceed\w*|record|raised?|strong\w*|improv\w*|accelerat\w*)\b"
NEGATIVE = r"\b(miss\w*|below|lower\w*|cut|weak\w*|declin\w*|headwind\w*)\b"


def lexicon_features(texts: list[str]) -> np.ndarray:
    rows = []
    for t in texts:
        pos, neg = len(re.findall(POSITIVE, t, re.I)), len(re.findall(NEGATIVE, t, re.I))
        rows.append([pos, neg, (pos - neg) / (pos + neg + 1)])
    return np.asarray(rows, float)


def _zfit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X.mean(0), X.std(0) + 1e-9


# ---------------------------------------------------------------------------
# Quant model: direction x magnitude
# ---------------------------------------------------------------------------


@dataclass
class QuantModel:
    """Ridge direction on the surprise-residualized target, LightGBM magnitude, product."""

    alpha: float = 3.0
    lex_weight: float = 0.5
    n_estimators: int = 200
    num_leaves: int = 7
    learning_rate: float = 0.05
    slope: float = 0.05  # affine map of z(d*m) into [0, 1]; the metric is invariant to it
    _fitted: dict = field(default_factory=dict, repr=False)

    def _direction_X(self, texts, fit: bool):
        if fit:
            self._fitted["tfidf"] = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            T = self._fitted["tfidf"].fit_transform(texts)
            L = lexicon_features(texts)
            self._fitted["lex"] = _zfit(L)
        else:
            T = self._fitted["tfidf"].transform(texts)
            L = lexicon_features(texts)
        mu, sd = self._fitted["lex"]
        return sp.hstack([T, sp.csr_matrix(self.lex_weight * (L - mu) / sd)]).tocsr()

    @staticmethod
    def _magnitude_X(timing: np.ndarray, texts: list[str]) -> np.ndarray:
        length = np.array([[len(t), len(re.findall(r"\d+(?:\.\d+)?%", t))] for t in texts], float)
        return np.column_stack([timing, length])

    def fit(self, texts: list[str], timing: np.ndarray, y: np.ndarray, surprise_pct: np.ndarray) -> QuantModel:
        y_res = residualize(y, surprise_pct)
        self._fitted["ridge"] = Ridge(alpha=self.alpha).fit(self._direction_X(texts, True), y_res)
        self._fitted["gbm"] = lgb.LGBMRegressor(
            n_estimators=self.n_estimators, num_leaves=self.num_leaves, learning_rate=self.learning_rate,
            min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0,
            verbose=-1).fit(self._magnitude_X(timing, texts), np.abs(np.asarray(y) - 0.5))
        dm = self.components(texts, timing)
        prod = dm[0] * dm[1]
        self._fitted["scale"] = (float(prod.mean()), float(prod.std() + 1e-12))
        return self

    def components(self, texts: list[str], timing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self._fitted["ridge"].predict(self._direction_X(texts, False))
        m = np.maximum(self._fitted["gbm"].predict(self._magnitude_X(timing, texts)), 1e-3)
        return d, m

    def predict(self, texts: list[str], timing: np.ndarray) -> np.ndarray:
        d, m = self.components(texts, timing)
        mu, sd = self._fitted["scale"]
        return np.clip(0.5 + self.slope * (d * m - mu) / sd, 0.01, 0.99)


# ---------------------------------------------------------------------------
# Point-in-time prior-surprise lookup
# ---------------------------------------------------------------------------


@dataclass
class PriorSurpriseTable:
    """Earlier events' surprise percentiles, admitted only if knowable at prediction time.

    An entry is usable for an event only if (a) it was released strictly before the
    event's BOUND and (b) its period's percentile ranks were FINAL before the bound
    (a rank within an unfinished period would use later events). The bound is the
    event's ``knowledge_cutoff`` when provided, else ``event_datetime - fallback_margin``,
    chosen to be no later than any true cutoff.
    """

    period_final_at: dict[str, datetime]
    fallback_margin: timedelta = timedelta(hours=120)
    _rows: dict[str, tuple[list[datetime], list[float], list[str]]] = field(default_factory=dict, repr=False)

    def add(self, ticker: str, released_at: datetime, surprise_pct: float, period: str) -> None:
        t, p, q = self._rows.setdefault(ticker, ([], [], []))
        i = bisect.bisect_right(t, released_at)
        t.insert(i, released_at)
        p.insert(i, float(surprise_pct))
        q.insert(i, period)

    def bound(self, event_datetime: datetime | None, knowledge_cutoff: datetime | None) -> datetime | None:
        if knowledge_cutoff is not None:
            return knowledge_cutoff
        return None if event_datetime is None else event_datetime - self.fallback_margin

    def lookup(self, ticker: str, event_datetime: datetime | None,
               knowledge_cutoff: datetime | None = None) -> float | None:
        bound = self.bound(event_datetime, knowledge_cutoff)
        rows = self._rows.get(ticker)
        if bound is None or rows is None:
            return None
        times, pcts, periods = rows
        for j in range(bisect.bisect_left(times, bound) - 1, -1, -1):
            final = self.period_final_at.get(periods[j])
            if final is not None and final < bound:
                return pcts[j]
        return None


# ---------------------------------------------------------------------------
# Prior-surprise adjustment with deterministic fallback
# ---------------------------------------------------------------------------


@dataclass
class PriorSurpriseAdjustment:
    """p = clip(base + gamma * (prev - mu) / sd); a missing ``prev`` returns ``base`` EXACTLY."""

    gamma: float = 0.0
    mu: float = 0.5
    sd: float = 0.29

    def fit(self, base_oof: np.ndarray, prev: np.ndarray, y_resid: np.ndarray, alpha: float = 1.0):
        """Ridge of the surprise-residualized target on z(base) and z(prev), rewritten on the
        base model's own scale. ``base_oof`` must be OUT-OF-SAMPLE base predictions."""
        base_oof, prev, y_resid = (np.asarray(a, float) for a in (base_oof, prev, y_resid))
        ok = ~np.isnan(prev)
        self.mu, self.sd = float(prev[ok].mean()), float(prev[ok].std() + 1e-12)
        mb, sb = base_oof.mean(), base_oof.std() + 1e-12
        Z = np.column_stack([(base_oof - mb) / sb, np.where(ok, (prev - self.mu) / self.sd, 0.0)])
        b = np.linalg.solve(Z.T @ Z + alpha * np.eye(2), Z.T @ (y_resid - y_resid.mean()))
        self.gamma = float(sb * b[1] / b[0]) if b[0] > 0 else 0.0
        return self

    def apply(self, base: float, prev: float | None) -> float:
        if prev is None or not np.isfinite(prev):
            return float(base)
        return float(np.clip(base + self.gamma * (prev - self.mu) / self.sd, 0.01, 0.99))
