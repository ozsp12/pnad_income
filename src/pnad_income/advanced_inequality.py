"""Extended inequality and Lorenz-shape measures for PNAD income data.

The routines in this module are deliberately unweighted.  They describe the
processed microdata records currently stored in the repository.  Population
inference from PNAD requires survey expansion weights and design information.

The Lorenz-based definitions follow the unit-square convention.  In particular,
the Kolkata index ``k`` is the solution of ``L(k) = 1-k``; the Pietra index is
the maximum vertical Lorenz gap; and the Zanardi index follows the geometric
construction revisited by Clementi et al. (2019), in which the discriminant point
splits the Lorenz curve into poor and rich concentration regions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .inequality import gini, lorenz_curve


def _values(values) -> np.ndarray:
    """Return finite nonnegative observations sorted in ascending order."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    return np.sort(x)


def pietra_index(values) -> float:
    """Return the Pietra (Robin Hood) index ``max[p-L(p)]``."""
    x = _values(values)
    if x.size == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0
    p, L = lorenz_curve(x)
    return float(np.max(p - L))


def kolkata_index(values) -> float:
    """Return the Kolkata index solving ``L(k)=1-k`` by linear interpolation."""
    x = _values(values)
    if x.size == 0:
        return np.nan
    if np.all(x == 0):
        return 0.5
    p, L = lorenz_curve(x)
    f = L + p - 1.0
    hi = int(np.searchsorted(f, 0.0, side="left"))
    if hi <= 0:
        return float(p[0])
    if hi >= len(p):
        return float(p[-1])
    lo = hi - 1
    if np.isclose(f[hi], f[lo]):
        return float(p[hi])
    weight = -f[lo] / (f[hi] - f[lo])
    return float(p[lo] + weight * (p[hi] - p[lo]))


def _insert_point(p: np.ndarray, L: np.ndarray, x0: float, y0: float) -> tuple[np.ndarray, np.ndarray]:
    """Insert a known point into a piecewise-linear Lorenz representation."""
    if np.any(np.isclose(p, x0, rtol=0.0, atol=1e-14)):
        idx = int(np.argmin(np.abs(p - x0)))
        out_p = p.copy()
        out_L = L.copy()
        out_p[idx] = x0
        out_L[idx] = y0
        return out_p, out_L
    idx = int(np.searchsorted(p, x0))
    return np.insert(p, idx, x0), np.insert(L, idx, y0)


def zanardi_components(values) -> dict[str, float]:
    """Return the Zanardi index and the geometric quantities used to construct it.

    The discriminant point is ``D=(p_d,q_d)`` with ``q_d=1-p_d`` and
    ``L(p_d)=q_d``.  The two sub-concentrations are obtained as the Lorenz-area
    deficits relative to the chords ``O-D`` and ``D-E``, normalized by
    ``K_d=p_d q_d/2``.  The Zanardi index is then

    ``Z_d = 2 K_d (G_r-G_p) / G``.
    """
    x = _values(values)
    if x.size == 0:
        return {
            "kolkata": np.nan,
            "qd": np.nan,
            "gini_poor": np.nan,
            "gini_rich": np.nan,
            "delta_gini": np.nan,
            "zanardi": np.nan,
        }
    if np.all(x == 0):
        return {
            "kolkata": 0.5,
            "qd": 0.5,
            "gini_poor": 0.0,
            "gini_rich": 0.0,
            "delta_gini": 0.0,
            "zanardi": 0.0,
        }

    G = float(gini(x))
    p, L = lorenz_curve(x)
    pd_ = kolkata_index(x)
    qd_ = 1.0 - pd_
    p2, L2 = _insert_point(p, L, pd_, qd_)
    split = int(np.argmin(np.abs(p2 - pd_)))

    p_left = p2[: split + 1]
    L_left = L2[: split + 1]
    chord_left = (qd_ / pd_) * p_left

    p_right = p2[split:]
    L_right = L2[split:]
    chord_right = qd_ + ((1.0 - qd_) / (1.0 - pd_)) * (p_right - pd_)

    Kd = pd_ * qd_ / 2.0
    if Kd <= 0:
        Gp = Gr = 0.0
    else:
        area_p = float(np.trapezoid(chord_left - L_left, p_left))
        area_r = float(np.trapezoid(chord_right - L_right, p_right))
        Gp = area_p / Kd
        Gr = area_r / Kd

    delta = Gr - Gp
    Z = 0.0 if np.isclose(G, 0.0) else (2.0 * Kd * delta / G)
    return {
        "kolkata": float(pd_),
        "qd": float(qd_),
        "gini_poor": float(Gp),
        "gini_rich": float(Gr),
        "delta_gini": float(delta),
        "zanardi": float(Z),
    }


def herfindahl_index(values) -> float:
    """Return the income-share Herfindahl index ``sum(s_i^2)``."""
    x = _values(values)
    total = x.sum()
    if x.size == 0 or total <= 0:
        return np.nan
    s = x / total
    return float(np.sum(s * s))


def normalized_herfindahl(values) -> float:
    """Return ``(N H - 1)/(N-1)`` so equal shares map to zero."""
    x = _values(values)
    if x.size == 0:
        return np.nan
    if x.size == 1:
        return 0.0
    H = herfindahl_index(x)
    if not np.isfinite(H):
        return np.nan
    return float((x.size * H - 1.0) / (x.size - 1.0))


def shannon_entropy(values) -> float:
    """Return Shannon entropy of the vector of income shares."""
    x = _values(values)
    total = x.sum()
    if x.size == 0 or total <= 0:
        return np.nan
    s = x / total
    positive = s[s > 0]
    return float(-np.sum(positive * np.log(positive)))


def normalized_shannon_inequality(values) -> float:
    """Return the entropy deficit ``1-S/log(N)``."""
    x = _values(values)
    if x.size < 2:
        return 0.0 if x.size == 1 else np.nan
    S = shannon_entropy(x)
    if not np.isfinite(S):
        return np.nan
    return float(1.0 - S / np.log(x.size))


def theil_index(values) -> float:
    """Return Theil T; zero incomes contribute the limiting value zero."""
    x = _values(values)
    if x.size == 0:
        return np.nan
    mean = x.mean()
    if mean <= 0:
        return 0.0
    ratio = x / mean
    positive = ratio > 0
    terms = ratio[positive] * np.log(ratio[positive])
    return float(np.sum(terms) / x.size)


def atkinson_index(values, epsilon: float = 0.5) -> float:
    """Return the Atkinson index for inequality-aversion parameter ``epsilon``."""
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    x = _values(values)
    if x.size == 0:
        return np.nan
    mean = x.mean()
    if mean <= 0:
        return 0.0
    if np.isclose(epsilon, 1.0):
        if np.any(x == 0):
            return 1.0
        ede = float(np.exp(np.mean(np.log(x))))
    else:
        power = 1.0 - epsilon
        if power < 0 and np.any(x == 0):
            return 1.0
        ede = float(np.mean(x ** power) ** (1.0 / power))
    return float(1.0 - ede / mean)


def annual_inequality_indices(
    df: pd.DataFrame,
    value_col: str = "income",
    year_col: str = "year",
    atkinson_epsilon: float = 0.5,
) -> pd.DataFrame:
    """Compute a longitudinal table of complementary inequality measures."""
    if value_col not in df.columns or year_col not in df.columns:
        raise KeyError(f"Required columns '{year_col}' and '{value_col}' are not both present.")

    rows: list[dict[str, float | int]] = []
    for year, group in df.groupby(year_col, sort=True):
        x = _values(pd.to_numeric(group[value_col], errors="coerce"))
        if x.size == 0:
            continue
        G = float(gini(x))
        P = pietra_index(x)
        z = zanardi_components(x)
        K = z["kolkata"]
        excess = 2.0 * K - 1.0 if np.isfinite(K) else np.nan
        rows.append({
            "year": int(year),
            "n": int(x.size),
            "gini": G,
            "pietra": P,
            "kolkata": K,
            "kolkata_excess": excess,
            "pietra_over_kolkata_excess": P / excess if excess > 0 else np.nan,
            "pietra_over_gini": P / G if G > 0 else np.nan,
            "kolkata_small_g_prediction": 0.5 + 0.375 * G,
            "pietra_small_g_prediction": 0.75 * G,
            "gini_poor": z["gini_poor"],
            "gini_rich": z["gini_rich"],
            "delta_gini": z["delta_gini"],
            "zanardi": z["zanardi"],
            "herfindahl": herfindahl_index(x),
            "herfindahl_normalized": normalized_herfindahl(x),
            "shannon_entropy": shannon_entropy(x),
            "shannon_inequality": normalized_shannon_inequality(x),
            "theil": theil_index(x),
            f"atkinson_{atkinson_epsilon:g}": atkinson_index(x, epsilon=atkinson_epsilon),
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
