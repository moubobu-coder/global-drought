"""Python implementation of Jeff Burkey ktaub.m (2020 revision).

Method: Jeff Burkey KTAUB (2020 revision).
Copyright (c) 2013, Jeff Burkey. All rights reserved.
License: BSD 3-Clause; see KTAUB_LICENSE.txt for the complete terms
and disclaimer. Keep that file with redistributed copies of this port.
The numerical method follows the
MATLAB source, including its S=0 p-value convention and interpolated slope CI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class KtaubResult:
    taub: float
    tau: float
    h: int
    p: float
    z: float
    s: int
    sigma: float
    sen: float
    n: int
    ci_lower: float
    ci_upper: float
    denominator_taub: float
    denominator_tau: float
    nsigma: float


def ktaub(datain: np.ndarray, alpha: float = 0.05) -> KtaubResult:
    """Return ktaub.m trend statistics for a two-column [time, value] array.

    The individual pairwise slopes are omitted because trendcal.m consumes
    only ``sen`` and ``h``. Use ``source_sen_line`` for the source's trend line.
    Returns the Sen slope, significance decision and Mann–Kendall statistics.
    """
    data = np.asarray(datain, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("datain must have shape (n, 2)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    # MATLAB: sorted = sortrows(datain,1); sorted(any(isnan(sorted),2),:) = []
    data = data[np.argsort(data[:, 0], kind="stable")]
    data = data[~np.isnan(data).any(axis=1)]
    n = len(data)
    if n < 3:
        raise ValueError("ktaub.m requires at least three valid observations")
    x, y = data.T

    # Literal MATLAB R2023b behavior: unique(sorted_values) gives indices of
    # the FIRST occurrence. ktaub.m diffs those indices, so with ties it does
    # not actually obtain all tie-run lengths (notably the final run). We keep
    # this source-code quirk rather than silently substituting a corrected MK.
    def source_tie_diffs(values: np.ndarray) -> np.ndarray:
        _, first = np.unique(np.sort(values), return_index=True)
        first = first + 1  # source uses 1-based indices
        return np.r_[max(int(first[0]), 1), np.diff(first)]

    tx = source_tie_diffs(x)
    ty = source_tie_diffs(y)
    tp = np.sum(tx * (tx - 1) * (2 * tx + 5))
    uq = np.sum(ty * (ty - 1) * (2 * ty + 5))
    tu1 = (np.sum(tx * (tx - 1) * (tx - 2))
           * np.sum(ty * (ty - 1) * (ty - 2))
           / (9 * n * (n - 1) * (n - 2)))
    tu2 = (np.sum(tx * (tx - 1)) * np.sum(ty * (ty - 1))
           / (2 * n * (n - 1)))
    t1a = np.sum(tx * (tx - 1)) / 2
    t2a = np.sum(ty * (ty - 1)) / 2

    # MATLAB triu pair matrices contain each chronological pair exactly once.
    i, j = np.triu_indices(n, k=1)
    dx = x[j] - x[i]
    dy = y[j] - y[i]
    s = int(np.sum(np.sign(dx) * np.sign(dy)))
    slopes = np.sort(dy[dx != 0] / dx[dx != 0])
    sen = float(np.median(slopes)) if len(slopes) else np.nan

    dall = n * (n - 1) / 2
    d = float(np.sqrt((dall - t1a) * (dall - t2a)))
    tau = s / dall
    taub = s / d if d > 0 else np.nan

    corrected_s = s - int(np.sign(s))
    sigma = float(np.sqrt((n * (n - 1) * (2 * n + 5) - tp - uq) / 18
                          + tu1 + tu2))
    nsigma = float(np.sqrt(n * (n - 1) * (2 * n + 5)))
    z = corrected_s / sigma if sigma > 0 else np.nan

    # MATLAB: M1=(Nprime-Calpha)/2; M2=(Nprime+Calpha)/2+1;
    # interp1q((1:Nprime),C3,M*) is linear interpolation on 1-based ranks.
    calpha = norm.ppf(1 - alpha / 2) * sigma
    nprime = len(slopes)
    m1 = (nprime - calpha) / 2
    m2 = (nprime + calpha) / 2 + 1
    ranks = np.arange(1, nprime + 1)
    ci_lower = float(np.interp(m1, ranks, slopes)) if 1 <= m1 <= nprime else np.nan
    ci_upper = float(np.interp(m2, ranks, slopes)) if 1 <= m2 <= nprime else np.nan

    # MATLAB ztest(s,0,sigma,alpha), except it substitutes 1 when corrected S=0.
    # The source says such a reported p should be read as a lower bound.
    p = float(2 * norm.sf(abs(1 if corrected_s == 0 else corrected_s) / sigma)) if sigma > 0 else np.nan
    h = int(p < alpha) if np.isfinite(p) else 0
    return KtaubResult(taub, tau, h, p, z, s, sigma, sen, n,
                       ci_lower, ci_upper, d, dall, nsigma)


def trendcal(data: np.ndarray, syear: int, eyear: int,
             landmask: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Port trendcal.m's (lon, lat, time) loop and 0-mask convention."""
    values = np.asarray(data)
    mask = np.asarray(landmask)
    years = np.arange(syear, eyear + 1, dtype=np.float64)
    if values.ndim != 3 or values.shape[:2] != mask.shape or values.shape[2] != len(years):
        raise ValueError("Expected data (lon, lat, time) and matching landmask")
    sen = np.full(mask.shape, np.nan)
    h = np.full(mask.shape, np.nan)
    for i, j in np.ndindex(mask.shape):
        if mask[i, j] == 0:
            continue
        result = ktaub(np.column_stack((years, values[i, j, :])), alpha)
        sen[i, j] = result.sen
        h[i, j] = result.h
    return sen, h


def source_sen_line(datain: np.ndarray, result: KtaubResult) -> np.ndarray:
    """Reproduce ktaub.m's unconditional ``senplot`` line placement.

    The MATLAB plotting block uses the unsorted input, its median y, and
    ``datain(round(length(datain)/2),1)`` as the line's time anchor. For 52
    annual points this anchors at observation 26, not the median year.
    """
    data = np.asarray(datain, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) == 0:
        raise ValueError("datain must have shape (n, 2) with n > 0")
    anchor_index = int(np.floor(len(data) / 2 + 0.5)) - 1
    line = np.median(data[:, 1]) + result.sen * (data[:, 0] - data[anchor_index, 0])
    return np.column_stack((data[:, 0], line))

