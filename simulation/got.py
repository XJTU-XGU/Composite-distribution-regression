r"""
One-dimensional GOT regression with one distributional predictor and multiple scalar predictors.

This file separates
  (A) simulation data generation,
  (B) scalar-to-noisy-distribution conversion, and
  (C) GOT fitting / ordering search / prediction.

The implementation follows the multiple-predictor Geodesic Optimal Transport
regression idea in the 1D Wasserstein space. Each distribution is represented
by its quantile vector on a common grid u_1,...,u_m.

Key model in this code
----------------------
For sample i and predictor j, the empirical transport map from the predictor
barycenter mu_j to X_{i,j} is represented by the monotone interpolation map

    T_{i,j}( Q_{mu_j}(u_l) ) = Q_{X_{i,j}}(u_l).

For alpha in a compact interval, the scaled map is approximated by the 1D
Wasserstein geodesic scaling

    T_{i,j}^{(alpha)}(x) = x + alpha * (T_{i,j}(x) - x).

For an ordered list order = [j_1,...,j_k], we use the paper convention

    (alpha_1 T_{j_1}) \oplus ... \oplus (alpha_k T_{j_k})
    = T_{j_1}^{(alpha_1)} \circ ... \circ T_{j_k}^{(alpha_k)}.

Therefore, when applying it to the response barycenter nu, the rightmost map
acts first. In code this means looping over reversed(order).

Notes
-----
1. The true alpha vector in the simulation satisfies sum_j alpha_j^2 = alpha_l2_sq,
   independent of p.
2. The default fitting uses unknown ordering and implements greedy sequential
   ordering search as in the paper: at step k, try each remaining predictor,
   optimize the k-dimensional alpha vector, and choose the one giving the
   smallest empirical W2 loss.
3. The reported time in run_one_replicate() is fit time + prediction/evaluation
   time, excluding data-generation time.

Author: ChatGPT generated helper code for experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from statistics import NormalDist
import time

import numpy as np

try:
    from scipy.optimize import minimize
except Exception as exc:  # pragma: no cover
    minimize = None


# ============================================================
# Basic quantile utilities
# ============================================================

def make_quantile_grid(m: int = 120, eps: float = 1e-4) -> np.ndarray:
    """Common quantile grid avoiding exactly 0 and 1."""
    return np.linspace(eps, 1.0 - eps, int(m))


def samples_to_quantiles(samples: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    """Convert samples to quantile vectors.

    Parameters
    ----------
    samples:
        Either shape (N, n_obs) or (N, p, n_obs).
    u_grid:
        Quantile levels in (0,1), shape (m,).
    """
    return np.quantile(np.asarray(samples, dtype=float), u_grid, axis=-1).transpose(
        *range(1, np.asarray(samples).ndim), 0
    )


def w2_sq_quantile(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Squared 1D W2 distance using quantile vectors.

    If q1, q2 have shape (..., m), returns shape (...,).
    """
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    return np.mean((q1 - q2) ** 2, axis=-1)


def _ensure_increasing(q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Make a vector nondecreasing with a tiny strictness for interpolation stability."""
    q = np.asarray(q, dtype=float).copy()
    q = np.maximum.accumulate(q)
    for k in range(1, len(q)):
        if q[k] <= q[k - 1]:
            q[k] = q[k - 1] + eps
    return q


_STD_NORMAL = NormalDist()


def _standard_normal_quantile_grid(u: np.ndarray) -> np.ndarray:
    """Standard normal quantiles evaluated on a grid in (0, 1)."""
    u = np.asarray(u, dtype=float)
    return np.array([_STD_NORMAL.inv_cdf(float(v)) for v in u], dtype=float)


def _interp_with_linear_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """1D interpolation with linear extrapolation outside the source quantile range.

    The original version used np.interp(..., left=fp[0], right=fp[-1]).  That clamps
    values outside the barycenter support.  For scalar predictors converted to
    Gaussian-like distributions, the induced transport map is essentially a shift,
    so linear extrapolation is much more stable when compositions move the response
    quantiles slightly outside the fitted barycenter range.
    """
    x_arr = np.asarray(x, dtype=float)
    xp = _ensure_increasing(np.asarray(xp, dtype=float))
    fp = _ensure_increasing(np.asarray(fp, dtype=float))

    y = np.interp(x_arr, xp, fp)
    if len(xp) >= 2:
        left = x_arr < xp[0]
        right = x_arr > xp[-1]
        slope_left = (fp[1] - fp[0]) / max(xp[1] - xp[0], 1e-12)
        slope_right = (fp[-1] - fp[-2]) / max(xp[-1] - xp[-2], 1e-12)
        y = np.asarray(y, dtype=float)
        y[left] = fp[0] + slope_left * (x_arr[left] - xp[0])
        y[right] = fp[-1] + slope_right * (x_arr[right] - xp[-1])
    return y


def scalar_predictors_to_noisy_distributions(
    C: np.ndarray,
    n_obs: int,
    scalar_noise_sd: float = 0.03,
    seed: int = 0,
    deterministic_noise: bool = True,
) -> np.ndarray:
    """Convert scalar predictors into empirical distributions by Gaussian smoothing.

    Parameters
    ----------
    C:
        Scalar predictor matrix, shape (N, d). If d=1, a vector of shape (N,)
        is also allowed.
    n_obs:
        Number of pseudo-observations used for each scalar-derived distribution.
    scalar_noise_sd:
        Standard deviation of the Gaussian perturbation.  Small values approximate
        degenerate scalar distributions; larger values make the induced densities
        smoother and interpolation more stable.
    seed:
        Used only when deterministic_noise=False.
    deterministic_noise:
        If True, use fixed Gaussian quantile points.  This removes Monte Carlo
        variability from the scalar-to-distribution conversion.  If False, use
        random Gaussian noise.

    Returns
    -------
    X_scalar:
        Array of shape (N, d, n_obs). Entry (i, j, :) is a pseudo-sample from
        N(C[i, j], scalar_noise_sd^2).
    """
    C = np.asarray(C, dtype=float)
    if C.ndim == 1:
        C = C[:, None]
    if C.ndim != 2:
        raise ValueError("C must have shape (N, d) or (N,).")
    if scalar_noise_sd < 0:
        raise ValueError("scalar_noise_sd must be nonnegative.")

    N, d = C.shape
    n_obs = int(n_obs)
    if n_obs <= 1:
        raise ValueError("n_obs must be greater than 1.")

    if scalar_noise_sd == 0:
        noise = np.zeros((N, d, n_obs), dtype=float)
    elif deterministic_noise:
        uu = (np.arange(n_obs, dtype=float) + 0.5) / n_obs
        z = _standard_normal_quantile_grid(uu)
        noise = scalar_noise_sd * z[None, None, :]
    else:
        rng = np.random.default_rng(seed)
        noise = scalar_noise_sd * rng.normal(size=(N, d, n_obs))

    return C[:, :, None] + noise


def build_mixed_predictor_array(
    X_distribution: np.ndarray,
    C_scalar: Optional[np.ndarray],
    scalar_noise_sd: float = 0.03,
    n_obs: Optional[int] = None,
    seed: int = 0,
    deterministic_noise: bool = True,
) -> np.ndarray:
    """Build the GOT input array from one distributional predictor and scalars.

    The returned array has the same format as the original all-distribution code:
    shape (N, p, n_obs), where predictor 0 is the original distributional predictor
    and predictors 1,...,p-1 are scalar predictors smoothed by Gaussian noise.
    """
    X_distribution = np.asarray(X_distribution, dtype=float)
    if X_distribution.ndim == 2:
        X0 = X_distribution[:, None, :]
    elif X_distribution.ndim == 3 and X_distribution.shape[1] == 1:
        X0 = X_distribution
    else:
        raise ValueError("X_distribution must have shape (N, n_obs) or (N, 1, n_obs).")

    N = X0.shape[0]
    n_obs_out = int(n_obs or X0.shape[-1])
    if n_obs_out != X0.shape[-1]:
        raise ValueError("For simplicity, n_obs must match X_distribution.shape[-1].")

    if C_scalar is None:
        return X0.copy()

    C_scalar = np.asarray(C_scalar, dtype=float)
    if C_scalar.ndim == 1:
        C_scalar = C_scalar[:, None]
    if C_scalar.shape[0] != N:
        raise ValueError("X_distribution and C_scalar must have the same N.")

    X_scalar = scalar_predictors_to_noisy_distributions(
        C_scalar,
        n_obs=n_obs_out,
        scalar_noise_sd=scalar_noise_sd,
        seed=seed,
        deterministic_noise=deterministic_noise,
    )
    return np.concatenate([X0, X_scalar], axis=1)


# ============================================================
# Simulation data generation
# ============================================================


def _random_zero_mean_sine_quantile_map(
    u_grid: np.ndarray,
    rng: np.random.Generator,
    amplitude: float = 0.35,
    n_basis: int = 4,
) -> np.ndarray:
    """Generate a random monotone quantile map with population barycenter id.

    We use

        Q(u) = u + sum_{r=1}^K c_r sin(pi r u)/(pi r),

    where the random coefficients are symmetric with sum |c_r| <= amplitude < 1.
    Then Q'(u) = 1 + sum_r c_r cos(pi r u) >= 1 - sum_r |c_r| > 0,
    so Q is monotone. Since E[c_r] = 0, E[Q(u)] = u for every u. Therefore the
    population W2 barycenter of these predictor distributions is Uniform[0,1].

    This is important for matching the GOT model used to generate responses:
    T_{mu_j, X_{i,j}} is exactly represented by Q_{X_{i,j}} when mu_j is uniform.
    """
    if not (0.0 <= amplitude < 1.0):
        raise ValueError("amplitude must be in [0, 1) to guarantee monotonicity.")
    u = np.asarray(u_grid, dtype=float)
    raw = rng.normal(size=int(n_basis))
    denom = np.sum(np.abs(raw)) + 1e-12
    radius = rng.uniform(0.0, float(amplitude))
    coeff = radius * raw / denom

    q = u.copy()
    for r, c in enumerate(coeff, start=1):
        q += c * np.sin(np.pi * r * u) / (np.pi * r)
    return _ensure_increasing(q)


def _random_positive_derivative_map(
    u_grid: np.ndarray,
    rng: np.random.Generator,
    amplitude: float = 0.35,
    max_freq: int = 4,
) -> np.ndarray:
    """Generate a random increasing map [0,1] -> [0,1] on u_grid.

    The map is generated by drawing a positive derivative v(u), then integrating
    and normalizing it. This keeps all generated predictor quantile functions
    monotone and stable.
    """
    u = np.asarray(u_grid, dtype=float)
    freq1 = int(rng.integers(1, max_freq + 1))
    freq2 = int(rng.integers(1, max_freq + 1))
    phase1 = rng.uniform(0.0, 2.0 * np.pi)
    phase2 = rng.uniform(0.0, 2.0 * np.pi)
    a1 = rng.normal(scale=amplitude)
    a2 = rng.normal(scale=0.5 * amplitude)

    log_v = a1 * np.sin(2.0 * np.pi * freq1 * u + phase1)
    log_v += a2 * np.cos(2.0 * np.pi * freq2 * u + phase2)
    v = np.exp(log_v)

    # Numerical integration of derivative.
    du = np.diff(u)
    increments = 0.5 * (v[:-1] + v[1:]) * du
    q = np.empty_like(u)
    q[0] = 0.0
    q[1:] = np.cumsum(increments)
    q = q / q[-1]

    # Rescale to approximately match u_grid endpoints instead of exact 0/1.
    q = u[0] + (u[-1] - u[0]) * q
    return _ensure_increasing(q)


def _apply_true_scaled_map(x: np.ndarray, u_grid: np.ndarray, map_grid: np.ndarray, alpha: float) -> np.ndarray:
    """Apply x + alpha * (T(x)-x), where T is represented on u_grid."""
    Tx = _interp_with_linear_extrap(x, u_grid, map_grid)
    return x + alpha * (Tx - x)


def _apply_true_scaled_map_from_barycenter(
    x: np.ndarray,
    q_source: np.ndarray,
    q_target: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Apply x + alpha * (T(x)-x) for T: q_source -> q_target."""
    Tx = _interp_with_linear_extrap(x, q_source, q_target)
    return x + alpha * (Tx - x)


def _sample_from_quantile(q_grid: np.ndarray, u_grid: np.ndarray, n_obs: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from a distribution represented by a quantile vector."""
    uu = rng.uniform(u_grid[0], u_grid[-1], size=int(n_obs))
    return np.interp(uu, u_grid, q_grid)


def make_true_alpha(
    p: int,
    alpha_l2_sq: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    kind: str = "dense_positive",
    sparsity: Optional[int] = None,
) -> np.ndarray:
    """Create true alpha with sum alpha_j^2 = alpha_l2_sq.

    Parameters
    ----------
    kind:
        "dense_positive" : all entries positive, random magnitudes.
        "equal_positive" : all entries equal sqrt(alpha_l2_sq / p).
        "sparse_positive": only `sparsity` entries are nonzero.
    """
    if rng is None:
        rng = np.random.default_rng()
    p = int(p)
    if p <= 0:
        raise ValueError("p must be positive.")
    if alpha_l2_sq <= 0:
        raise ValueError("alpha_l2_sq must be positive.")

    if kind == "equal_positive":
        alpha = np.ones(p) / np.sqrt(p)
    elif kind == "sparse_positive":
        s = int(sparsity or max(1, min(p, int(np.sqrt(p)))))
        s = max(1, min(p, s))
        alpha = np.zeros(p)
        active = rng.choice(p, size=s, replace=False)
        vals = rng.uniform(0.5, 1.5, size=s)
        alpha[active] = vals
    elif kind == "dense_positive":
        alpha = rng.uniform(0.5, 1.5, size=p)
    else:
        raise ValueError(f"Unknown alpha kind: {kind}")

    alpha = alpha / np.linalg.norm(alpha) * np.sqrt(alpha_l2_sq)
    return alpha


def generate_got_1d_data(
    N_train: int = 100,
    N_test: int = 100,
    p: int = 20,
    n_obs: int = 100,
    m_true: int = 500,
    alpha_l2_sq: float = 0.5,
    alpha_kind: str = "dense_positive",
    sparsity: Optional[int] = None,
    true_order: Optional[Sequence[int]] = None,
    random_true_order: bool = False,
    map_amplitude: float = 0.35,
    noise_sd: float = 0.02,
    seed: int = 2026,
) -> Dict[str, np.ndarray]:
    """Generate a synthetic 1D multiple-distribution-predictor GOT dataset.

    Returns a dictionary with sample arrays and latent quantile-level objects.

    Shapes
    ------
    X_train: (N_train, p, n_obs)
    Y_train: (N_train, n_obs)
    X_test : (N_test, p, n_obs)
    Y_test : (N_test, n_obs)
    alpha_true: (p,)
    true_order: (p,)
    """
    rng = np.random.default_rng(seed)
    p = int(p)
    u_grid = make_quantile_grid(m_true)

    alpha_true = make_true_alpha(
        p=p,
        alpha_l2_sq=alpha_l2_sq,
        rng=rng,
        kind=alpha_kind,
        sparsity=sparsity,
    )

    if true_order is None:
        if random_true_order:
            true_order_arr = rng.permutation(p)
        else:
            true_order_arr = np.arange(p)
    else:
        true_order_arr = np.asarray(true_order, dtype=int)
        if sorted(true_order_arr.tolist()) != list(range(p)):
            raise ValueError("true_order must be a permutation of 0,...,p-1.")

    q_nu_true = u_grid.copy()  # response barycenter: Uniform[0,1]

    def generate_split(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = np.empty((N, p, n_obs), dtype=float)
        Y = np.empty((N, n_obs), dtype=float)
        qX_all = np.empty((N, p, m_true), dtype=float)
        qY_all = np.empty((N, m_true), dtype=float)

        for i in range(N):
            maps = []
            for j in range(p):
                qX = _random_zero_mean_sine_quantile_map(
                    u_grid=u_grid,
                    rng=rng,
                    amplitude=map_amplitude,
                )
                qX_all[i, j] = qX
                maps.append(qX)
                X[i, j] = _sample_from_quantile(qX, u_grid, n_obs, rng)

            qY = q_nu_true.copy()
            # Paper convention: T_{j_1} \circ ... \circ T_{j_p}; rightmost map acts first.
            for j in reversed(true_order_arr):
                qY = _apply_true_scaled_map(
                    qY,
                    u_grid=u_grid,
                    map_grid=maps[j],
                    alpha=float(alpha_true[j]),
                )

            if noise_sd > 0:
                # Add smooth monotone-ish response noise through a small positive-derivative map.
                noise_map = _random_zero_mean_sine_quantile_map(
                    u_grid=u_grid,
                    rng=rng,
                    amplitude=noise_sd,
                    n_basis=3,
                )
                qY = _apply_true_scaled_map(qY, u_grid, noise_map, alpha=1.0)

            qY = _ensure_increasing(np.clip(qY, u_grid[0], u_grid[-1]))
            qY_all[i] = qY
            Y[i] = _sample_from_quantile(qY, u_grid, n_obs, rng)

        return X, Y, qX_all, qY_all

    X_train, Y_train, qX_train_true, qY_train_true = generate_split(int(N_train))
    X_test, Y_test, qX_test_true, qY_test_true = generate_split(int(N_test))

    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test,
        "alpha_true": alpha_true,
        "true_order": true_order_arr,
        "u_grid_true": u_grid,
        "q_nu_true": q_nu_true,
        "qX_train_true": qX_train_true,
        "qY_train_true": qY_train_true,
        "qX_test_true": qX_test_true,
        "qY_test_true": qY_test_true,
        "config": {
            "N_train": N_train,
            "N_test": N_test,
            "p": p,
            "n_obs": n_obs,
            "m_true": m_true,
            "alpha_l2_sq": alpha_l2_sq,
            "alpha_kind": alpha_kind,
            "sparsity": sparsity,
            "map_amplitude": map_amplitude,
            "noise_sd": noise_sd,
            "seed": seed,
        },
    }



# ============================================================
# Mixed simulation: predictor 0 is a distribution; predictors 1,...,p-1 are scalars
# ============================================================


def generate_mixed_got_1d_data(
    N_train: int = 100,
    N_test: int = 100,
    p: int = 20,
    n_obs: int = 100,
    m_true: int = 500,
    alpha_l2_sq: float = 0.5,
    alpha_kind: str = "dense_positive",
    sparsity: Optional[int] = None,
    true_order: Optional[Sequence[int]] = None,
    random_true_order: bool = False,
    map_amplitude: float = 0.35,
    response_noise_sd: float = 0.02,
    scalar_noise_sd: float = 0.03,
    scalar_range: Tuple[float, float] = (-0.5, 0.5),
    deterministic_scalar_noise: bool = True,
    seed: int = 2026,
) -> Dict[str, np.ndarray]:
    """Generate data with one distribution predictor and p-1 scalar predictors.

    Predictor 0 is genuinely distributional. Predictors 1,...,p-1 are generated
    as scalars C_j and then converted into noisy empirical distributions

        C_j + scalar_noise_sd * Z,     Z ~ N(0, 1).

    The response is generated by applying the same GOT composition idea after
    converting each scalar into a Gaussian-smoothed distribution. Therefore the
    downstream fitting code can still use the original input format (N, p, n_obs).

    Returned raw objects include both the expanded GOT input X_* and the original
    pieces X_*_dist and C_*_scalar.
    """
    rng = np.random.default_rng(seed)
    p = int(p)
    if p < 1:
        raise ValueError("p must be at least 1. Predictor 0 is the distributional predictor.")
    d_scalar = p - 1
    u_grid = make_quantile_grid(m_true)
    z_grid = _standard_normal_quantile_grid(u_grid)

    alpha_true = make_true_alpha(
        p=p,
        alpha_l2_sq=alpha_l2_sq,
        rng=rng,
        kind=alpha_kind,
        sparsity=sparsity,
    )

    if true_order is None:
        if random_true_order:
            true_order_arr = rng.permutation(p)
        else:
            true_order_arr = np.arange(p)
    else:
        true_order_arr = np.asarray(true_order, dtype=int)
        if sorted(true_order_arr.tolist()) != list(range(p)):
            raise ValueError("true_order must be a permutation of 0,...,p-1.")

    q_nu_true = u_grid.copy()  # response barycenter used as the starting template
    scalar_center = 0.5 * (float(scalar_range[0]) + float(scalar_range[1]))
    q_mu_true = np.empty((p, m_true), dtype=float)
    q_mu_true[0] = u_grid.copy()
    for j in range(1, p):
        q_mu_true[j] = scalar_center + float(scalar_noise_sd) * z_grid
        q_mu_true[j] = _ensure_increasing(q_mu_true[j])

    def _scalar_quantile(c: float) -> np.ndarray:
        return _ensure_increasing(float(c) + float(scalar_noise_sd) * z_grid)

    def generate_split(N: int, split_seed_offset: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_dist = np.empty((N, n_obs), dtype=float)
        C_scalar = np.empty((N, d_scalar), dtype=float)
        X = np.empty((N, p, n_obs), dtype=float)
        Y = np.empty((N, n_obs), dtype=float)
        qX_all = np.empty((N, p, m_true), dtype=float)
        qY_all = np.empty((N, m_true), dtype=float)

        scalar_noise_seed = int(seed + split_seed_offset)

        for i in range(N):
            maps = []

            # Predictor 0: a genuine distributional predictor.
            qX0 = _random_zero_mean_sine_quantile_map(
                u_grid=u_grid,
                rng=rng,
                amplitude=map_amplitude,
            )
            qX_all[i, 0] = qX0
            maps.append(qX0)
            X_dist[i] = _sample_from_quantile(qX0, u_grid, n_obs, rng)
            X[i, 0] = X_dist[i]

            # Predictors 1,...,p-1: scalars, then Gaussian smoothing.
            if d_scalar > 0:
                C_scalar[i] = rng.uniform(float(scalar_range[0]), float(scalar_range[1]), size=d_scalar)
                for k in range(d_scalar):
                    c = float(C_scalar[i, k])
                    qC = _scalar_quantile(c)
                    j = k + 1
                    qX_all[i, j] = qC
                    maps.append(qC)

            # Use the same conversion routine for the observed/fitted predictors.
            if d_scalar > 0:
                X_scalar_i = scalar_predictors_to_noisy_distributions(
                    C_scalar[i : i + 1],
                    n_obs=n_obs,
                    scalar_noise_sd=scalar_noise_sd,
                    seed=scalar_noise_seed + i,
                    deterministic_noise=deterministic_scalar_noise,
                )[0]
                X[i, 1:] = X_scalar_i

            qY = q_nu_true.copy()
            # Paper convention: T_{j_1} o ... o T_{j_p}; rightmost map acts first.
            for j in reversed(true_order_arr):
                qY = _apply_true_scaled_map_from_barycenter(
                    qY,
                    q_source=q_mu_true[int(j)],
                    q_target=maps[int(j)],
                    alpha=float(alpha_true[int(j)]),
                )

            if response_noise_sd > 0:
                noise_map = _random_zero_mean_sine_quantile_map(
                    u_grid=u_grid,
                    rng=rng,
                    amplitude=response_noise_sd,
                    n_basis=3,
                )
                qY = _apply_true_scaled_map_from_barycenter(qY, u_grid, noise_map, alpha=1.0)

            qY = _ensure_increasing(qY)
            qY_all[i] = qY
            Y[i] = _sample_from_quantile(qY, u_grid, n_obs, rng)

        return X, Y, X_dist, C_scalar, qX_all, qY_all

    X_train, Y_train, X_train_dist, C_train_scalar, qX_train_true, qY_train_true = generate_split(int(N_train), 1000000)
    X_test, Y_test, X_test_dist, C_test_scalar, qX_test_true, qY_test_true = generate_split(int(N_test), 2000000)

    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test,
        "X_train_dist": X_train_dist,
        "X_test_dist": X_test_dist,
        "C_train_scalar": C_train_scalar,
        "C_test_scalar": C_test_scalar,
        "alpha_true": alpha_true,
        "true_order": true_order_arr,
        "u_grid_true": u_grid,
        "q_nu_true": q_nu_true,
        "q_mu_true": q_mu_true,
        "qX_train_true": qX_train_true,
        "qY_train_true": qY_train_true,
        "qX_test_true": qX_test_true,
        "qY_test_true": qY_test_true,
        "predictor_types": np.array(["distribution"] + ["scalar_gaussian"] * d_scalar, dtype=object),
        "config": {
            "N_train": N_train,
            "N_test": N_test,
            "p": p,
            "n_obs": n_obs,
            "m_true": m_true,
            "alpha_l2_sq": alpha_l2_sq,
            "alpha_kind": alpha_kind,
            "sparsity": sparsity,
            "map_amplitude": map_amplitude,
            "response_noise_sd": response_noise_sd,
            "scalar_noise_sd": scalar_noise_sd,
            "scalar_range": scalar_range,
            "deterministic_scalar_noise": deterministic_scalar_noise,
            "seed": seed,
        },
    }


# ============================================================
# GOT model with greedy ordering search
# ============================================================

@dataclass
class GOT1DRegressor:
    """One-dimensional multiple-predictor GOT regression.

    Parameters
    ----------
    m:
        Number of quantile grid points used for fitting.
    fit_order:
        If True, use greedy sequential ordering search. If False, use `order`.
    order:
        Fixed order used when fit_order=False.
    alpha_bounds:
        Compact parameter set K. For this simulation true alphas are positive,
        so (0, 1) is usually appropriate.
    order_maxiter:
        Max optimizer iterations for each candidate during ordering search.
    final_maxiter:
        Max optimizer iterations for final alpha refit after order selection.
    verbose:
        Print progress during order search.
    """

    m: int = 120
    fit_order: bool = True
    order: Optional[Sequence[int]] = None
    alpha_bounds: Tuple[float, float] = (0.0, 1.0)
    order_maxiter: int = 80
    final_maxiter: int = 300
    optimizer_method: str = "L-BFGS-B"
    verbose: bool = False

    u_grid_: Optional[np.ndarray] = field(default=None, init=False)
    qX_: Optional[np.ndarray] = field(default=None, init=False)
    qY_: Optional[np.ndarray] = field(default=None, init=False)
    q_mu_: Optional[np.ndarray] = field(default=None, init=False)
    q_nu_: Optional[np.ndarray] = field(default=None, init=False)
    order_: Optional[np.ndarray] = field(default=None, init=False)
    alpha_: Optional[np.ndarray] = field(default=None, init=False)
    fit_info_: Dict[str, object] = field(default_factory=dict, init=False)

    def _check_optimizer(self) -> None:
        if minimize is None:
            raise ImportError("scipy is required for optimization. Please install scipy.")

    def _prepare_quantiles(self, X: np.ndarray, Y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.ndim != 3:
            raise ValueError("X must have shape (N, p, n_obs).")
        if Y.ndim != 2:
            raise ValueError("Y must have shape (N, n_obs).")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples.")

        self.u_grid_ = make_quantile_grid(self.m)
        self.qX_ = samples_to_quantiles(X, self.u_grid_)  # (N, p, m)
        self.qY_ = samples_to_quantiles(Y, self.u_grid_)  # (N, m)
        self.q_mu_ = np.mean(self.qX_, axis=0)             # (p, m)
        self.q_nu_ = np.mean(self.qY_, axis=0)             # (m,)

        # Stabilize interpolation grids.
        for j in range(self.q_mu_.shape[0]):
            self.q_mu_[j] = _ensure_increasing(self.q_mu_[j])
        self.q_nu_ = _ensure_increasing(self.q_nu_)

    @property
    def n_predictors_(self) -> int:
        if self.qX_ is None:
            raise RuntimeError("Model has not been fitted.")
        return int(self.qX_.shape[1])

    def _apply_scaled_map_train(self, i: int, j: int, x: np.ndarray, alpha: float) -> np.ndarray:
        assert self.qX_ is not None and self.q_mu_ is not None
        q_mu_j = self.q_mu_[j]
        q_x_ij = self.qX_[i, j]
        Tx = _interp_with_linear_extrap(x, q_mu_j, q_x_ij)
        return x + alpha * (Tx - x)

    def _predict_train_quantile_one(self, i: int, order: Sequence[int], alpha: Sequence[float]) -> np.ndarray:
        assert self.q_nu_ is not None
        pred = self.q_nu_.copy()
        # Paper convention: left-to-right composition, so apply reversed order to the object.
        for j, a in zip(reversed(order), reversed(alpha)):
            pred = self._apply_scaled_map_train(i, int(j), pred, float(a))
        return pred

    def _loss_for_order_alpha(self, order: Sequence[int], alpha: Sequence[float]) -> float:
        assert self.qY_ is not None
        N = self.qY_.shape[0]
        loss = 0.0
        for i in range(N):
            pred = self._predict_train_quantile_one(i, order, alpha)
            loss += float(np.mean((self.qY_[i] - pred) ** 2))
        return loss / N

    def _optimize_alpha(
        self,
        order: Sequence[int],
        alpha_init: Optional[np.ndarray] = None,
        maxiter: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        self._check_optimizer()
        k = len(order)
        lo, hi = self.alpha_bounds
        if alpha_init is None:
            alpha_init = np.full(k, min(max(0.05, lo), hi), dtype=float)
        else:
            alpha_init = np.asarray(alpha_init, dtype=float)
            if alpha_init.shape[0] != k:
                raise ValueError("alpha_init has wrong length.")
            alpha_init = np.clip(alpha_init, lo, hi)

        bounds = [(lo, hi)] * k

        def obj(a: np.ndarray) -> float:
            return self._loss_for_order_alpha(order, a)

        res = minimize(
            obj,
            alpha_init,
            method=self.optimizer_method,
            bounds=bounds,
            options={"maxiter": int(maxiter or self.final_maxiter), "ftol": 1e-10},
        )
        alpha_hat = np.asarray(res.x, dtype=float)
        loss = float(res.fun)
        return alpha_hat, loss

    def _greedy_select_order(self) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
        p = self.n_predictors_
        remaining = list(range(p))
        selected: List[int] = []
        alpha_selected = np.empty(0, dtype=float)
        history: List[Dict[str, object]] = []

        for step in range(p):
            best = None
            for j in remaining:
                candidate_order = selected + [j]
                if step == 0:
                    init = np.array([0.05], dtype=float)
                else:
                    init = np.concatenate([alpha_selected, np.array([0.05])])

                alpha_hat, loss = self._optimize_alpha(
                    candidate_order,
                    alpha_init=init,
                    maxiter=self.order_maxiter,
                )
                item = {
                    "candidate": int(j),
                    "order": candidate_order,
                    "alpha": alpha_hat,
                    "loss": loss,
                }
                if best is None or loss < best["loss"]:
                    best = item

            assert best is not None
            selected = list(best["order"])
            alpha_selected = np.asarray(best["alpha"], dtype=float)
            remaining.remove(int(best["candidate"]))
            history.append(best)

            if self.verbose:
                print(
                    f"[order search] step={step + 1:03d}/{p}, "
                    f"selected={best['candidate']}, loss={best['loss']:.6e}"
                )

        return np.asarray(selected, dtype=int), alpha_selected, history

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "GOT1DRegressor":
        """Fit GOT regression. If fit_order=True, unknown order is greedily searched."""
        t0 = time.perf_counter()
        self._prepare_quantiles(X, Y)
        p = self.n_predictors_

        if self.fit_order:
            order_hat, alpha_init, history = self._greedy_select_order()
            # Final refit over the selected full order.
            alpha_hat, final_loss = self._optimize_alpha(
                order_hat,
                alpha_init=alpha_init,
                maxiter=self.final_maxiter,
            )
        else:
            if self.order is None:
                order_hat = np.arange(p)
            else:
                order_hat = np.asarray(self.order, dtype=int)
            if sorted(order_hat.tolist()) != list(range(p)):
                raise ValueError("order must be a permutation of 0,...,p-1.")
            history = []
            alpha_hat, final_loss = self._optimize_alpha(
                order_hat,
                alpha_init=None,
                maxiter=self.final_maxiter,
            )

        self.order_ = np.asarray(order_hat, dtype=int)
        self.alpha_ = np.asarray(alpha_hat, dtype=float)
        self.fit_info_ = {
            "fit_time": time.perf_counter() - t0,
            "training_loss": final_loss,
            "order_history": history,
        }
        return self

    def _apply_scaled_map_test(
        self,
        q_x_one: np.ndarray,
        j: int,
        x: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        assert self.q_mu_ is not None
        q_mu_j = self.q_mu_[j]
        q_x_j = q_x_one[j]
        Tx = _interp_with_linear_extrap(x, q_mu_j, q_x_j)
        return x + alpha * (Tx - x)

    def predict_quantiles(self, X: np.ndarray) -> np.ndarray:
        """Predict response quantile vectors for X.

        Parameters
        ----------
        X: shape (N_test, p, n_obs)
        Returns
        -------
        q_pred: shape (N_test, m)
        """
        if self.order_ is None or self.alpha_ is None or self.q_nu_ is None:
            raise RuntimeError("Model has not been fitted.")
        X = np.asarray(X, dtype=float)
        qX_test = samples_to_quantiles(X, self.u_grid_)  # type: ignore[arg-type]
        N_test = qX_test.shape[0]
        q_pred = np.empty((N_test, self.m), dtype=float)
        for i in range(N_test):
            pred = self.q_nu_.copy()
            for j, a in zip(reversed(self.order_), reversed(self.alpha_)):
                pred = self._apply_scaled_map_test(qX_test[i], int(j), pred, float(a))
            q_pred[i] = _ensure_increasing(pred)
        return q_pred

    def predict_samples(self, X: np.ndarray, n_pred: Optional[int] = None, seed: int = 0) -> np.ndarray:
        """Generate samples from predicted response distributions."""
        rng = np.random.default_rng(seed)
        q_pred = self.predict_quantiles(X)
        n_pred = int(n_pred or X.shape[-1])
        out = np.empty((q_pred.shape[0], n_pred), dtype=float)
        for i in range(q_pred.shape[0]):
            out[i] = _sample_from_quantile(q_pred[i], self.u_grid_, n_pred, rng)  # type: ignore[arg-type]
        return out



@dataclass
class GOT1DMixedRegressor:
    """Wrapper for one distributional predictor plus multiple scalar predictors.

    Use this class when your raw input is (X_distribution, C_scalar), where
    X_distribution has shape (N, n_obs) and C_scalar has shape (N, d).  Internally,
    C_scalar is converted to Gaussian-smoothed empirical distributions and the
    original GOT1DRegressor is used unchanged afterwards.
    """

    m: int = 120
    scalar_noise_sd: float = 0.03
    deterministic_scalar_noise: bool = True
    scalar_seed: int = 12345
    fit_order: bool = True
    order: Optional[Sequence[int]] = None
    alpha_bounds: Tuple[float, float] = (0.0, 1.0)
    order_maxiter: int = 80
    final_maxiter: int = 300
    optimizer_method: str = "L-BFGS-B"
    verbose: bool = False

    base_model_: Optional[GOT1DRegressor] = field(default=None, init=False)
    n_obs_: Optional[int] = field(default=None, init=False)
    n_scalar_: Optional[int] = field(default=None, init=False)

    @property
    def order_(self) -> Optional[np.ndarray]:
        return None if self.base_model_ is None else self.base_model_.order_

    @property
    def alpha_(self) -> Optional[np.ndarray]:
        return None if self.base_model_ is None else self.base_model_.alpha_

    @property
    def fit_info_(self) -> Dict[str, object]:
        return {} if self.base_model_ is None else self.base_model_.fit_info_

    @property
    def u_grid_(self) -> Optional[np.ndarray]:
        return None if self.base_model_ is None else self.base_model_.u_grid_

    def _build_X(self, X_distribution: np.ndarray, C_scalar: Optional[np.ndarray], seed: int) -> np.ndarray:
        return build_mixed_predictor_array(
            X_distribution=X_distribution,
            C_scalar=C_scalar,
            scalar_noise_sd=self.scalar_noise_sd,
            n_obs=self.n_obs_,
            seed=seed,
            deterministic_noise=self.deterministic_scalar_noise,
        )

    def fit(self, X_distribution: np.ndarray, C_scalar: Optional[np.ndarray], Y: np.ndarray) -> "GOT1DMixedRegressor":
        X_distribution = np.asarray(X_distribution, dtype=float)
        if X_distribution.ndim != 2:
            raise ValueError("X_distribution must have shape (N, n_obs).")
        self.n_obs_ = int(X_distribution.shape[-1])
        if C_scalar is None:
            self.n_scalar_ = 0
        else:
            C_tmp = np.asarray(C_scalar, dtype=float)
            if C_tmp.ndim == 1:
                C_tmp = C_tmp[:, None]
            self.n_scalar_ = int(C_tmp.shape[1])

        X = self._build_X(X_distribution, C_scalar, seed=self.scalar_seed)
        self.base_model_ = GOT1DRegressor(
            m=self.m,
            fit_order=self.fit_order,
            order=self.order,
            alpha_bounds=self.alpha_bounds,
            order_maxiter=self.order_maxiter,
            final_maxiter=self.final_maxiter,
            optimizer_method=self.optimizer_method,
            verbose=self.verbose,
        )
        self.base_model_.fit(X, Y)
        return self

    def predict_quantiles(self, X_distribution: np.ndarray, C_scalar: Optional[np.ndarray]) -> np.ndarray:
        if self.base_model_ is None:
            raise RuntimeError("Model has not been fitted.")
        X = self._build_X(X_distribution, C_scalar, seed=self.scalar_seed + 999999)
        return self.base_model_.predict_quantiles(X)

    def predict_samples(
        self,
        X_distribution: np.ndarray,
        C_scalar: Optional[np.ndarray],
        n_pred: Optional[int] = None,
        seed: int = 0,
    ) -> np.ndarray:
        if self.base_model_ is None:
            raise RuntimeError("Model has not been fitted.")
        rng = np.random.default_rng(seed)
        q_pred = self.predict_quantiles(X_distribution, C_scalar)
        n_pred = int(n_pred or self.n_obs_ or X_distribution.shape[-1])
        out = np.empty((q_pred.shape[0], n_pred), dtype=float)
        for i in range(q_pred.shape[0]):
            out[i] = _sample_from_quantile(q_pred[i], self.base_model_.u_grid_, n_pred, rng)  # type: ignore[arg-type]
        return out


def evaluate_prediction_w2(
    model: GOT1DRegressor,
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> Dict[str, float]:
    """Predict and compute average W2 prediction errors on test data."""
    t0 = time.perf_counter()
    q_pred = model.predict_quantiles(X_test)
    qY_test = samples_to_quantiles(np.asarray(Y_test, dtype=float), model.u_grid_)  # type: ignore[arg-type]
    w2_sq = w2_sq_quantile(q_pred, qY_test)
    test_time = time.perf_counter() - t0
    return {
        "mean_w2_sq": float(np.mean(w2_sq)),
        "std_w2_sq": float(np.std(w2_sq)),
        "mean_w2": float(np.mean(np.sqrt(w2_sq))),
        "std_w2": float(np.std(np.sqrt(w2_sq))),
        "test_time": float(test_time),
    }




def evaluate_mixed_prediction_w2(
    model: GOT1DMixedRegressor,
    X_test_dist: np.ndarray,
    C_test_scalar: Optional[np.ndarray],
    Y_test: np.ndarray,
) -> Dict[str, float]:
    """Predict and compute average W2 errors for the mixed raw input format."""
    if model.base_model_ is None:
        raise RuntimeError("Model has not been fitted.")
    t0 = time.perf_counter()
    q_pred = model.predict_quantiles(X_test_dist, C_test_scalar)
    qY_test = samples_to_quantiles(np.asarray(Y_test, dtype=float), model.base_model_.u_grid_)  # type: ignore[arg-type]
    w2_sq = w2_sq_quantile(q_pred, qY_test)
    test_time = time.perf_counter() - t0
    return {
        "mean_w2_sq": float(np.mean(w2_sq)),
        "std_w2_sq": float(np.std(w2_sq)),
        "mean_w2": float(np.mean(np.sqrt(w2_sq))),
        "std_w2": float(np.std(np.sqrt(w2_sq))),
        "test_time": float(test_time),
    }


# ============================================================
# Experiment wrappers: unknown ordering + train/test total time
# ============================================================

def order_recovery_metrics(order_hat: Sequence[int], true_order: Sequence[int]) -> Dict[str, float]:
    """Simple order recovery diagnostics."""
    order_hat = np.asarray(order_hat, dtype=int)
    true_order = np.asarray(true_order, dtype=int)
    p = len(true_order)
    exact = float(np.array_equal(order_hat, true_order))
    pos_true = np.empty(p, dtype=int)
    pos_hat = np.empty(p, dtype=int)
    for k, j in enumerate(true_order):
        pos_true[j] = k
    for k, j in enumerate(order_hat):
        pos_hat[j] = k
    mean_abs_rank_error = float(np.mean(np.abs(pos_hat - pos_true)))
    top1_correct = float(order_hat[0] == true_order[0])
    top5_set_correct = float(set(order_hat[: min(5, p)]) == set(true_order[: min(5, p)]))
    return {
        "order_exact": exact,
        "top1_correct": top1_correct,
        "top5_set_correct": top5_set_correct,
        "mean_abs_rank_error": mean_abs_rank_error,
    }


def run_one_replicate(
    p: int = 20,
    N_train: int = 100,
    N_test: int = 100,
    n_obs: int = 100,
    m_fit: int = 120,
    alpha_l2_sq: float = 0.5,
    alpha_kind: str = "dense_positive",
    sparsity: Optional[int] = None,
    random_true_order: bool = False,
    noise_sd: float = 0.02,
    map_amplitude: float = 0.35,
    seed: int = 2026,
    fit_order: bool = True,
    alpha_bounds: Tuple[float, float] = (0.0, 1.0),
    order_maxiter: int = 80,
    final_maxiter: int = 300,
    verbose: bool = False,
) -> Dict[str, object]:
    """Generate one dataset, fit GOT, and report train+test time and prediction error.

    The reported `total_train_test_time` excludes data generation time.
    """
    data = generate_got_1d_data(
        N_train=N_train,
        N_test=N_test,
        p=p,
        n_obs=n_obs,
        alpha_l2_sq=alpha_l2_sq,
        alpha_kind=alpha_kind,
        sparsity=sparsity,
        random_true_order=random_true_order,
        noise_sd=noise_sd,
        map_amplitude=map_amplitude,
        seed=seed,
    )

    model = GOT1DRegressor(
        m=m_fit,
        fit_order=fit_order,
        order=None if fit_order else list(range(p)),
        alpha_bounds=alpha_bounds,
        order_maxiter=order_maxiter,
        final_maxiter=final_maxiter,
        verbose=verbose,
    )

    model.fit(data["X_train"], data["Y_train"])
    metrics = evaluate_prediction_w2(model, data["X_test"], data["Y_test"])

    fit_time = float(model.fit_info_["fit_time"])
    test_time = float(metrics["test_time"])
    total_time = fit_time + test_time

    alpha_true = np.asarray(data["alpha_true"], dtype=float)
    # Compare alpha by predictor identity, not by selected order position.
    alpha_hat_by_predictor = np.zeros(p, dtype=float)
    for pos, j in enumerate(model.order_):  # type: ignore[union-attr]
        alpha_hat_by_predictor[int(j)] = model.alpha_[pos]  # type: ignore[index]

    out: Dict[str, object] = {
        "p": int(p),
        "N_train": int(N_train),
        "N_test": int(N_test),
        "n_obs": int(n_obs),
        "m_fit": int(m_fit),
        "seed": int(seed),
        "fit_order": bool(fit_order),
        "mean_w2": metrics["mean_w2"],
        "mean_w2_sq": metrics["mean_w2_sq"],
        "fit_time": fit_time,
        "test_time": test_time,
        "total_train_test_time": total_time,
        "training_loss": model.fit_info_["training_loss"],
        "alpha_l2_sq_true": float(np.sum(alpha_true ** 2)),
        "alpha_l2_sq_hat": float(np.sum(model.alpha_ ** 2)),  # type: ignore[arg-type]
        "alpha_rmse_by_predictor": float(np.sqrt(np.mean((alpha_hat_by_predictor - alpha_true) ** 2))),
        "true_order": data["true_order"],
        "order_hat": model.order_,
        "alpha_true": alpha_true,
        "alpha_hat_in_order": model.alpha_,
        "alpha_hat_by_predictor": alpha_hat_by_predictor,
    }
    out.update(order_recovery_metrics(model.order_, data["true_order"]))  # type: ignore[arg-type]
    return out




def run_one_mixed_replicate(
    p: int = 20,
    N_train: int = 100,
    N_test: int = 100,
    n_obs: int = 100,
    m_fit: int = 120,
    alpha_l2_sq: float = 0.5,
    alpha_kind: str = "dense_positive",
    sparsity: Optional[int] = None,
    random_true_order: bool = False,
    response_noise_sd: float = 0.02,
    map_amplitude: float = 0.35,
    scalar_noise_sd: float = 0.03,
    scalar_range: Tuple[float, float] = (-0.5, 0.5),
    deterministic_scalar_noise: bool = True,
    seed: int = 2026,
    fit_order: bool = True,
    alpha_bounds: Tuple[float, float] = (0.0, 1.0),
    order_maxiter: int = 80,
    final_maxiter: int = 300,
    verbose: bool = False,
) -> Dict[str, object]:
    """One replicate for the mixed setting: predictor 0 distribution, others scalar."""
    data = generate_mixed_got_1d_data(
        N_train=N_train,
        N_test=N_test,
        p=p,
        n_obs=n_obs,
        alpha_l2_sq=alpha_l2_sq,
        alpha_kind=alpha_kind,
        sparsity=sparsity,
        random_true_order=random_true_order,
        response_noise_sd=response_noise_sd,
        map_amplitude=map_amplitude,
        scalar_noise_sd=scalar_noise_sd,
        scalar_range=scalar_range,
        deterministic_scalar_noise=deterministic_scalar_noise,
        seed=seed,
    )

    model = GOT1DMixedRegressor(
        m=m_fit,
        scalar_noise_sd=scalar_noise_sd,
        deterministic_scalar_noise=deterministic_scalar_noise,
        scalar_seed=seed + 777,
        fit_order=fit_order,
        order=None if fit_order else list(range(p)),
        alpha_bounds=alpha_bounds,
        order_maxiter=order_maxiter,
        final_maxiter=final_maxiter,
        verbose=verbose,
    )

    model.fit(data["X_train_dist"], data["C_train_scalar"], data["Y_train"])
    metrics = evaluate_mixed_prediction_w2(
        model,
        data["X_test_dist"],
        data["C_test_scalar"],
        data["Y_test"],
    )

    fit_time = float(model.fit_info_["fit_time"])
    test_time = float(metrics["test_time"])
    total_time = fit_time + test_time

    alpha_true = np.asarray(data["alpha_true"], dtype=float)
    alpha_hat = np.asarray(model.alpha_, dtype=float)
    order_hat = np.asarray(model.order_, dtype=int)
    alpha_hat_by_predictor = np.zeros(p, dtype=float)
    for pos, j in enumerate(order_hat):
        alpha_hat_by_predictor[int(j)] = alpha_hat[pos]

    out: Dict[str, object] = {
        "p": int(p),
        "n_scalar": int(p - 1),
        "N_train": int(N_train),
        "N_test": int(N_test),
        "n_obs": int(n_obs),
        "m_fit": int(m_fit),
        "seed": int(seed),
        "fit_order": bool(fit_order),
        "mean_w2": metrics["mean_w2"],
        "mean_w2_sq": metrics["mean_w2_sq"],
        "fit_time": fit_time,
        "test_time": test_time,
        "total_train_test_time": total_time,
        "training_loss": model.fit_info_["training_loss"],
        "scalar_noise_sd": float(scalar_noise_sd),
        "alpha_l2_sq_true": float(np.sum(alpha_true ** 2)),
        "alpha_l2_sq_hat": float(np.sum(alpha_hat ** 2)),
        "alpha_rmse_by_predictor": float(np.sqrt(np.mean((alpha_hat_by_predictor - alpha_true) ** 2))),
        "true_order": data["true_order"],
        "order_hat": order_hat,
        "alpha_true": alpha_true,
        "alpha_hat_in_order": alpha_hat,
        "alpha_hat_by_predictor": alpha_hat_by_predictor,
        "predictor_types": data["predictor_types"],
    }
    out.update(order_recovery_metrics(order_hat, data["true_order"]))
    return out


def run_p_grid_experiment(
    p_list: Sequence[int] = (5, 10, 20, 50, 100),
    reps: int = 10,
    N_train: int = 100,
    N_test: int = 100,
    n_obs: int = 100,
    m_fit: int = 120,
    alpha_l2_sq: float = 0.5,
    alpha_kind: str = "dense_positive",
    sparsity: Optional[int] = None,
    random_true_order: bool = False,
    noise_sd: float = 0.02,
    map_amplitude: float = 0.35,
    base_seed: int = 2026,
    fit_order: bool = True,
    alpha_bounds: Tuple[float, float] = (0.0, 1.0),
    order_maxiter: int = 80,
    final_maxiter: int = 300,
    verbose: bool = False,
):
    """Run several p values and return a pandas DataFrame if pandas is available."""
    rows: List[Dict[str, object]] = []
    for p in p_list:
        for r in range(int(reps)):
            seed = int(base_seed + 100000 * int(p) + r)
            out = run_one_replicate(
                p=int(p),
                N_train=N_train,
                N_test=N_test,
                n_obs=n_obs,
                m_fit=m_fit,
                alpha_l2_sq=alpha_l2_sq,
                alpha_kind=alpha_kind,
                sparsity=sparsity,
                random_true_order=random_true_order,
                noise_sd=noise_sd,
                map_amplitude=map_amplitude,
                seed=seed,
                fit_order=fit_order,
                alpha_bounds=alpha_bounds,
                order_maxiter=order_maxiter,
                final_maxiter=final_maxiter,
                verbose=verbose,
            )
            rows.append(out)
            print(
                f"[p={p:3d}, rep={r + 1:3d}/{reps}] "
                f"W2={out['mean_w2']:.4e}, "
                f"W2sq={out['mean_w2_sq']:.4e}, "
                f"time={out['total_train_test_time']:.2f}s, "
                f"order_exact={out['order_exact']:.0f}"
            )

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        summary = (
            df.groupby("p")[[
                "mean_w2",
                "mean_w2_sq",
                "fit_time",
                "test_time",
                "total_train_test_time",
                "alpha_rmse_by_predictor",
                "order_exact",
                "top1_correct",
                "top5_set_correct",
                "mean_abs_rank_error",
            ]]
            .agg(["mean", "std"])
            .reset_index()
        )
        print("\nSummary by p:")
        print(summary)
        return df, summary
    except Exception:
        return rows, None




def run_p_grid_mixed_experiment(
    p_list: Sequence[int] = (5, 10, 20, 50, 100),
    reps: int = 10,
    N_train: int = 100,
    N_test: int = 100,
    n_obs: int = 100,
    m_fit: int = 120,
    alpha_l2_sq: float = 0.5,
    alpha_kind: str = "dense_positive",
    sparsity: Optional[int] = None,
    random_true_order: bool = False,
    response_noise_sd: float = 0.02,
    map_amplitude: float = 0.35,
    scalar_noise_sd: float = 0.03,
    scalar_range: Tuple[float, float] = (-0.5, 0.5),
    deterministic_scalar_noise: bool = True,
    base_seed: int = 2026,
    fit_order: bool = True,
    alpha_bounds: Tuple[float, float] = (0.0, 1.0),
    order_maxiter: int = 80,
    final_maxiter: int = 300,
    verbose: bool = False,
):
    """Run mixed experiments over several total predictor counts p."""
    rows: List[Dict[str, object]] = []
    for p in p_list:
        for r in range(int(reps)):
            seed = int(base_seed + 100000 * int(p) + r)
            out = run_one_mixed_replicate(
                p=int(p),
                N_train=N_train,
                N_test=N_test,
                n_obs=n_obs,
                m_fit=m_fit,
                alpha_l2_sq=alpha_l2_sq,
                alpha_kind=alpha_kind,
                sparsity=sparsity,
                random_true_order=random_true_order,
                response_noise_sd=response_noise_sd,
                map_amplitude=map_amplitude,
                scalar_noise_sd=scalar_noise_sd,
                scalar_range=scalar_range,
                deterministic_scalar_noise=deterministic_scalar_noise,
                seed=seed,
                fit_order=fit_order,
                alpha_bounds=alpha_bounds,
                order_maxiter=order_maxiter,
                final_maxiter=final_maxiter,
                verbose=verbose,
            )
            rows.append(out)
            print(
                f"[mixed p={p:3d}, rep={r + 1:3d}/{reps}] "
                f"W2={out['mean_w2']:.4e}, "
                f"W2sq={out['mean_w2_sq']:.4e}, "
                f"time={out['total_train_test_time']:.2f}s, "
                f"order_exact={out['order_exact']:.0f}"
            )

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        summary = (
            df.groupby("p")[[
                "mean_w2",
                "mean_w2_sq",
                "fit_time",
                "test_time",
                "total_train_test_time",
                "alpha_rmse_by_predictor",
                "order_exact",
                "top1_correct",
                "top5_set_correct",
                "mean_abs_rank_error",
            ]]
            .agg(["mean", "std"])
            .reset_index()
        )
        print("\nMixed summary by p:")
        print(summary)
        return df, summary
    except Exception:
        return rows, None


if __name__ == "__main__":
    # A small smoke test for the mixed setting:
    # predictor 0 is distributional; predictors 1,...,p-1 are scalars converted
    # to Gaussian-smoothed empirical distributions.
    df, summary = run_p_grid_mixed_experiment(
        p_list=(3,5,10),
        reps=5,
        N_train=1000,
        N_test=1000,
        n_obs=1000,
        m_fit=80,
        alpha_l2_sq=1.0,
        scalar_noise_sd=0.03,
        fit_order=True,
        order_maxiter=100,
        final_maxiter=100,
        verbose=False,
    )
