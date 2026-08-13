#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
VERSION: friedman10_structured_transport_pgfr_v1

Compare two mixed-predictor distribution-regression methods under a nonlinear
Friedman-type data-generating mechanism with one distributional predictor and
10 scalar predictors.

Data setting
------------
For each observation i,

    X_i ~ mu_i,
    C_i = (C_{i1}, ..., C_{i10}) in [0, 1]^10,
    Y_i = S_eps(S_0(X_i; C_i)),

where mu_i is a random mixture of three Beta distributions and

    S_0(x; C) = phi0(C) * zeta_4(x) + phi1(C).

Let the classical five-variable Friedman function be

    F(z1,...,z5)
      = 10 sin(pi z1 z2) + 20 (z3 - 0.5)^2 + 10 z4 + 5 z5,

and define H=(F-15)/15, whose range is contained in [-1,1]. We use

    phi0(C) = 1 + 0.3 H(C1,...,C5),
    phi1(C) =     0.3 H(C6,...,C10).

Thus phi0 depends only on C1,...,C5 and phi1 depends only on C6,...,C10.
Each component is sparse with respect to the full 10-dimensional scalar
covariate, while all 10 scalar predictors affect the complete transport map.
Moreover, phi0(C) lies in [0.7,1.3], so S_0 remains increasing in x.

Important modeling point
------------------------
The structured-transport estimator is still fitted with its original linear
covariate specification. Therefore, this script evaluates prediction under
nonlinear covariate-effect misspecification; it is not a correctly specified
convergence-rate experiment.

Methods
-------
1. structured_transport
   Calls fig4.prepare_grid_data and fig4.fit_ours_cvec_from_grid.

2. TW2025_PGFR
   Calls partially_global_frechet.fit_pgfr_baseline_from_samples.

Output
------
The script saves a torch-serialized .mat file, matching the existing
comparison scripts' convention, with the following structure:

    results["p_11_N_100"][method] = {
        "w1":   [mean test W1 for each successful replicate],
        "time": total training-plus-testing time across successful replicates,
    }

The "p_11" prefix is retained for compatibility with scripts in which p
denotes one distributional predictor plus 10 scalar predictors.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


# ============================================================
# Robust local imports
# ============================================================

def _load_phi_module():
    """Load Structured_transport_map_core.py robustly, including uploaded names with suffixes."""
    try:
        import Structured_transport_map_core  # type: ignore
        return Structured_transport_map_core
    except Exception:
        pass

    candidates = [
        _THIS_DIR / "Structured_transport_map_core.py",
        _THIS_DIR / "phi_0_1(1).py",
        _THIS_DIR / "phi_0_1(2).py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("phi_0_1_loaded", str(path))
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            sys.modules.setdefault("phi_0_1", module)
            return module

    raise ImportError(
        "Could not locate Structured_transport_map_core.py or phi_0_1(...).py in the script folder."
    )


phi = _load_phi_module()
sys.modules.setdefault("phi_0_1", phi)


def _load_fig4_module():
    """Load base_utils.py robustly and expose its estimator helpers."""
    try:
        import base_utils  # type: ignore
        return base_utils
    except Exception:
        pass

    path = _THIS_DIR / "base_utils.py"
    if path.exists():
        spec = importlib.util.spec_from_file_location("fig4_loaded", str(path))
        if spec is None or spec.loader is None:
            raise ImportError("Could not load base_utils.py.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module

    raise ImportError("Could not locate base_utils.py in the script folder.")


fig4_mod = _load_fig4_module()
fig4_prepare_grid_data = fig4_mod.prepare_grid_data
fig4_fit_ours_cvec_from_grid = fig4_mod.fit_ours_cvec_from_grid

try:
    from partially_global_frechet import fit_pgfr_baseline_from_samples
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Could not import partially_global_frechet.py. "
        "Please put this script in the same folder as that file."
    ) from exc


# ============================================================
# Basic utilities
# ============================================================

def _as_2d_c(C: Optional[np.ndarray], n: Optional[int] = None) -> np.ndarray:
    """Return scalar covariates as an array of shape (N, d)."""
    if C is None:
        if n is None:
            raise ValueError("Need n when C is None.")
        return np.zeros((int(n), 0), dtype=float)

    C = np.asarray(C, dtype=float)
    if C.ndim == 1:
        C = C[:, None]
    if C.ndim != 2:
        raise ValueError("C must have shape (N, d) or (N,).")
    return C


def _ensure_increasing(q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Make a vector nondecreasing with tiny strictness for interpolation."""
    q = np.asarray(q, dtype=float).copy()
    q = np.maximum.accumulate(q)
    for k in range(1, len(q)):
        if q[k] <= q[k - 1]:
            q[k] = q[k - 1] + eps
    return q


def _safe_jsonable(x):
    """Convert common NumPy/Python objects into serializable values."""
    if x is None:
        return ""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _safe_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_safe_jsonable(v) for v in x]
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def make_quantile_grid(m: int) -> np.ndarray:
    """Use midpoint quantile levels to avoid the endpoints 0 and 1."""
    m = int(m)
    if m <= 0:
        raise ValueError("m must be positive.")
    return (np.arange(m, dtype=float) + 0.5) / float(m)


def _quantiles_from_samples(X: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    """Convert an (N,n) sample matrix to an (N,m) quantile matrix."""
    X = np.asarray(X, dtype=float)
    u_grid = np.asarray(u_grid, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (N, n_obs).")
    return np.quantile(X, u_grid, axis=1).T


def quantile_w1(q_pred: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """Approximate one-dimensional W1 by averaging quantile differences."""
    q_pred = np.asarray(q_pred, dtype=float)
    q_true = np.asarray(q_true, dtype=float)
    if q_pred.shape != q_true.shape:
        raise ValueError(
            f"q_pred and q_true shapes differ: {q_pred.shape} vs {q_true.shape}"
        )
    return np.mean(np.abs(q_pred - q_true), axis=-1)


# ============================================================
# Friedman-type nonlinear covariate effects
# ============================================================

N_SCALAR = 10
TOTAL_PREDICTOR_COUNT = 11  # one distributional predictor + 10 scalars
RESULT_KEY = f"p_{TOTAL_PREDICTOR_COUNT}"
METHOD_NAMES = (
    "structured_transport",
    "TW2025_PGFR",
)


def friedman1(Z: np.ndarray) -> np.ndarray:
    """Evaluate the classical five-variable Friedman regression function.

    Parameters
    ----------
    Z : array_like, shape (..., 5)
        Covariates taking values in [0,1].
    """
    Z = np.asarray(Z, dtype=float)
    if Z.shape[-1] != 5:
        raise ValueError(f"friedman1 expects a last dimension of 5, got {Z.shape}.")

    z1, z2, z3, z4, z5 = np.moveaxis(Z, -1, 0)
    return (
        10.0 * np.sin(np.pi * z1 * z2)
        + 20.0 * (z3 - 0.5) ** 2
        + 10.0 * z4
        + 5.0 * z5
    )


def scaled_friedman1(Z: np.ndarray) -> np.ndarray:
    """Scale Friedman-1 from [0,30] to a subset of [-1,1]."""
    return (friedman1(Z) - 15.0) / 15.0


def evaluate_true_phi(
    C: np.ndarray,
    phi0_center: float = 1.0,
    phi0_amplitude: float = 0.3,
    phi1_center: float = 0.0,
    phi1_amplitude: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate nonlinear phi0 and phi1 for a matrix of 10 scalar predictors."""
    C = np.asarray(C, dtype=float)
    was_1d = C.ndim == 1
    if was_1d:
        C = C[None, :]
    if C.ndim != 2 or C.shape[1] != N_SCALAR:
        raise ValueError(f"C must have shape (N,{N_SCALAR}) or ({N_SCALAR},).")

    h0 = scaled_friedman1(C[:, :5])
    h1 = scaled_friedman1(C[:, 5:])

    phi0 = float(phi0_center) + float(phi0_amplitude) * h0
    phi1 = float(phi1_center) + float(phi1_amplitude) * h1

    if np.any(phi0 <= 0):
        raise ValueError(
            "phi0 must remain positive to preserve monotonicity in x. "
            "Reduce phi0_amplitude or increase phi0_center."
        )

    if was_1d:
        return phi0[0], phi1[0]
    return phi0, phi1


def S0_true_friedman(
    x: np.ndarray,
    C: np.ndarray,
    phi0_center: float = 1.0,
    phi0_amplitude: float = 0.3,
    phi1_center: float = 0.0,
    phi1_amplitude: float = 0.3,
) -> np.ndarray:
    """True map S0(x;C)=phi0(C) zeta_4(x)+phi1(C)."""
    x = np.asarray(x, dtype=float)
    phi0_value, phi1_value = evaluate_true_phi(
        C,
        phi0_center=phi0_center,
        phi0_amplitude=phi0_amplitude,
        phi1_center=phi1_center,
        phi1_amplitude=phi1_amplitude,
    )
    return float(phi0_value) * phi.zeta_k(x, k=4) + float(phi1_value)


def _theoretical_s0_bounds(
    phi0_center: float,
    phi0_amplitude: float,
    phi1_center: float,
    phi1_amplitude: float,
) -> Tuple[float, float]:
    """Conservative S0 bounds using H in [-1,1] and zeta_4([0,1])=[0,1]."""
    phi0_min = float(phi0_center) - abs(float(phi0_amplitude))
    phi0_max = float(phi0_center) + abs(float(phi0_amplitude))
    phi1_min = float(phi1_center) - abs(float(phi1_amplitude))
    phi1_max = float(phi1_center) + abs(float(phi1_amplitude))

    if phi0_min <= 0:
        raise ValueError(
            "Require phi0_center - abs(phi0_amplitude) > 0 to preserve monotonicity."
        )

    # Since zeta_4(x) is in [0,1] and phi0 is positive:
    s0_lower = phi1_min
    s0_upper = phi0_max + phi1_max
    return s0_lower, s0_upper


# ============================================================
# Data generation
# ============================================================

def generate_friedman10_data(
    N_train: int = 100,
    N_test: int = 1000,
    n_obs: int = 100,
    pis: Tuple[float, float, float] = (0.3, 0.4, 0.3),
    J_segments: int = 8,
    phi0_center: float = 1.0,
    phi0_amplitude: float = 0.3,
    phi1_center: float = 0.0,
    phi1_amplitude: float = 0.3,
    domain_min: float = 0.0,
    domain_max: float = 4.0,
    domain_buffer: float = 0.25,
    seed: int = 2026,
) -> Dict[str, object]:
    """Generate train/test data with 10-dimensional Friedman covariate effects."""
    # Structured_transport_map_core.py uses NumPy's global random state internally, while C is drawn
    # from a local generator. Setting both makes each replicate reproducible.
    np.random.seed(int(seed))
    rng = np.random.default_rng(int(seed))

    if int(N_train) <= 0 or int(N_test) <= 0 or int(n_obs) <= 0:
        raise ValueError("N_train, N_test, and n_obs must all be positive.")
    if len(pis) != 3 or not np.isclose(sum(pis), 1.0):
        raise ValueError("pis must contain three mixture weights summing to 1.")

    s0_lower, s0_upper = _theoretical_s0_bounds(
        phi0_center=phi0_center,
        phi0_amplitude=phi0_amplitude,
        phi1_center=phi1_center,
        phi1_amplitude=phi1_amplitude,
    )
    domain_min_used = min(float(domain_min), float(s0_lower - domain_buffer))
    domain_max_used = max(float(domain_max), float(s0_upper + domain_buffer))

    def _make_split(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        X = np.empty((int(N), int(n_obs)), dtype=float)
        Y = np.empty((int(N), int(n_obs)), dtype=float)
        C = rng.uniform(0.0, 1.0, size=(int(N), N_SCALAR))

        phi0_values, phi1_values = evaluate_true_phi(
            C,
            phi0_center=phi0_center,
            phi0_amplitude=phi0_amplitude,
            phi1_center=phi1_center,
            phi1_amplitude=phi1_amplitude,
        )

        observed_s0_min = np.inf
        observed_s0_max = -np.inf

        for i in range(int(N)):
            alphas, betas = phi.sample_mu_params()
            Xi = phi.sample_from_mixture_beta(alphas, betas, pis, size=int(n_obs))
            S_eps = phi.sample_random_S_eps(
                J=int(J_segments),
                domain_min=float(domain_min_used),
                domain_max=float(domain_max_used),
            )

            S0_values = S0_true_friedman(
                Xi,
                C[i],
                phi0_center=phi0_center,
                phi0_amplitude=phi0_amplitude,
                phi1_center=phi1_center,
                phi1_amplitude=phi1_amplitude,
            )
            if np.min(S0_values) < domain_min_used - 1e-10:
                raise RuntimeError("Generated S0 values fell below the noise-map domain.")
            if np.max(S0_values) > domain_max_used + 1e-10:
                raise RuntimeError("Generated S0 values exceeded the noise-map domain.")

            Yi = S_eps(S0_values)
            X[i, :] = np.asarray(Xi, dtype=float)
            Y[i, :] = np.asarray(Yi, dtype=float)

            observed_s0_min = min(observed_s0_min, float(np.min(S0_values)))
            observed_s0_max = max(observed_s0_max, float(np.max(S0_values)))

        diagnostics = {
            "phi0_min": float(np.min(phi0_values)),
            "phi0_max": float(np.max(phi0_values)),
            "phi1_min": float(np.min(phi1_values)),
            "phi1_max": float(np.max(phi1_values)),
            "s0_min": float(observed_s0_min),
            "s0_max": float(observed_s0_max),
        }
        return X, Y, C, diagnostics

    X_train, Y_train, C_train, train_diag = _make_split(int(N_train))
    X_test, Y_test, C_test, test_diag = _make_split(int(N_test))

    return {
        "X_train_dist": X_train,
        "Y_train": Y_train,
        "C_train_scalar": C_train,
        "X_test_dist": X_test,
        "Y_test": Y_test,
        "C_test_scalar": C_test,
        "config": {
            "data_source": "friedman10_phi0_c1_c5_phi1_c6_c10",
            "N_train": int(N_train),
            "N_test": int(N_test),
            "n_obs": int(n_obs),
            "n_distribution_predictors": 1,
            "n_scalar": N_SCALAR,
            "total_predictor_count": TOTAL_PREDICTOR_COUNT,
            "scalar_distribution": "iid_uniform_0_1",
            "pis": list(pis),
            "J_segments": int(J_segments),
            "phi0_definition": "phi0_center + phi0_amplitude * (F(C1:C5)-15)/15",
            "phi1_definition": "phi1_center + phi1_amplitude * (F(C6:C10)-15)/15",
            "phi0_center": float(phi0_center),
            "phi0_amplitude": float(phi0_amplitude),
            "phi1_center": float(phi1_center),
            "phi1_amplitude": float(phi1_amplitude),
            "theoretical_s0_lower": float(s0_lower),
            "theoretical_s0_upper": float(s0_upper),
            "domain_min_requested": float(domain_min),
            "domain_max_requested": float(domain_max),
            "domain_min_used": float(domain_min_used),
            "domain_max_used": float(domain_max_used),
            "domain_buffer": float(domain_buffer),
            "model_specification": "nonlinear_truth_linear_structured_transport_fit",
            "seed": int(seed),
            "train_diagnostics": train_diag,
            "test_diagnostics": test_diag,
        },
    }


# ============================================================
# Method 1: structured transport map
# ============================================================

@dataclass
class Fig4StructuredTransportRegressor:
    """Structured transport map using base_utils.py's existing implementation."""

    m_bins: int = 80
    max_iter: int = 80
    tol: float = 1e-6
    ridge: float = 1e-8
    enforce_phi0_positive: bool = True
    enforce_monotone_prediction: bool = True

    centers_: Optional[np.ndarray] = field(default=None, init=False)
    params_: Optional[Dict[str, object]] = field(default=None, init=False)
    S_hat_func_: Optional[object] = field(default=None, init=False)
    n_scalar_: Optional[int] = field(default=None, init=False)

    def fit(self, X_dist: np.ndarray, C: np.ndarray, Y: np.ndarray):
        C = _as_2d_c(C, n=np.asarray(X_dist).shape[0])
        if C.shape[1] != N_SCALAR:
            raise ValueError(f"Expected {N_SCALAR} scalar predictors, got {C.shape[1]}.")
        self.n_scalar_ = int(C.shape[1])

        X_list = [np.asarray(x, dtype=float) for x in np.asarray(X_dist)]
        Y_list = [np.asarray(y, dtype=float) for y in np.asarray(Y)]

        centers, w_ij, y_ij = fig4_prepare_grid_data(
            X_list=X_list,
            Y_list=Y_list,
            m_bins=self.m_bins,
        )
        params, S_hat = fig4_fit_ours_cvec_from_grid(
            centers=centers,
            w_ij=w_ij,
            y_ij=y_ij,
            C_list=C,
            max_iter=self.max_iter,
            tol=self.tol,
            ridge=self.ridge,
            enforce_phi0_positive=self.enforce_phi0_positive,
        )

        self.centers_ = np.asarray(centers, dtype=float)
        self.params_ = params
        self.S_hat_func_ = S_hat
        return self

    def predict_quantiles(
        self,
        X_dist: np.ndarray,
        C: np.ndarray,
        u_grid_eval: np.ndarray,
    ) -> np.ndarray:
        if self.S_hat_func_ is None or self.n_scalar_ is None:
            raise RuntimeError("Model not fitted.")

        X_dist = np.asarray(X_dist, dtype=float)
        C = _as_2d_c(C, n=X_dist.shape[0])
        if C.shape[1] != self.n_scalar_:
            raise ValueError(f"Expected {self.n_scalar_} scalar covariates, got {C.shape[1]}.")

        qX = _quantiles_from_samples(X_dist, u_grid_eval)
        q_pred = np.empty_like(qX, dtype=float)
        for i in range(qX.shape[0]):
            q = np.asarray(self.S_hat_func_(qX[i], C[i]), dtype=float)
            if self.enforce_monotone_prediction:
                q = _ensure_increasing(q)
            q_pred[i] = q
        return q_pred

    def info(self) -> Dict[str, object]:
        return {
            "m_bins": self.m_bins,
            "source": "fig4.fit_ours_cvec_from_grid",
            "n_scalar": self.n_scalar_,
            "params": _safe_jsonable(self.params_),
        }


# ============================================================
# Method 2: partially-global Frechet regression
# ============================================================

@dataclass
class ExternalTuckerWuPGFRRegressor:
    """Wrapper around partially_global_frechet.fit_pgfr_baseline_from_samples."""

    n_quantiles: int = 120
    bandwidth: Optional[float] = None
    bandwidth_scale: float = 1.0
    quantile_eps: float = 1e-4
    ridge: float = 1e-8
    pred_sample_size: Optional[int] = None

    predictor_: Optional[object] = field(default=None, init=False)
    info_: Optional[Dict[str, object]] = field(default=None, init=False)

    def fit(self, X_dist: np.ndarray, C: np.ndarray, Y: np.ndarray):
        C = _as_2d_c(C, n=np.asarray(X_dist).shape[0])
        if C.shape[1] != N_SCALAR:
            raise ValueError(f"Expected {N_SCALAR} scalar predictors, got {C.shape[1]}.")

        X_list = [np.asarray(x, dtype=float) for x in np.asarray(X_dist)]
        Y_list = [np.asarray(y, dtype=float) for y in np.asarray(Y)]
        self.predictor_, self.info_ = fit_pgfr_baseline_from_samples(
            X_list=X_list,
            Y_list=Y_list,
            c_list=C,
            bandwidth=self.bandwidth,
            bandwidth_scale=self.bandwidth_scale,
            n_quantiles=self.n_quantiles,
            quantile_eps=self.quantile_eps,
            ridge=self.ridge,
        )
        return self

    def predict_quantiles(
        self,
        X_dist: np.ndarray,
        C: np.ndarray,
        u_grid_eval: np.ndarray,
    ) -> np.ndarray:
        if self.predictor_ is None or self.info_ is None:
            raise RuntimeError("External PGFR predictor not fitted.")

        X_dist = np.asarray(X_dist, dtype=float)
        C = _as_2d_c(C, n=X_dist.shape[0])
        if C.shape[1] != N_SCALAR:
            raise ValueError(f"Expected {N_SCALAR} scalar predictors, got {C.shape[1]}.")

        n_pred = int(self.pred_sample_size or X_dist.shape[1])
        q_out = np.empty((X_dist.shape[0], len(u_grid_eval)), dtype=float)
        for i in range(X_dist.shape[0]):
            y_pred_sample = self.predictor_(X_dist[i], C[i], n_pred=n_pred)
            q_out[i] = np.quantile(
                np.asarray(y_pred_sample, dtype=float),
                u_grid_eval,
            )
        return q_out

    def info(self) -> Dict[str, object]:
        if self.info_ is None:
            return {}
        return {
            "bandwidth": float(self.info_.get("bandwidth", np.nan)),
            "bandwidth_scale": self.bandwidth_scale,
            "covariate_dim": int(self.info_.get("covariate_dim", -1)),
            "external_pgfr_call": True,
            "pred_sample_size": self.pred_sample_size,
            "source": "partially_global_frechet.fit_pgfr_baseline_from_samples",
        }


# ============================================================
# Evaluation
# ============================================================

def fit_predict_score_method(
    method_name: str,
    model,
    data: Dict[str, object],
    u_grid_eval: np.ndarray,
) -> Dict[str, object]:
    """Fit one method, predict, and return replicate-level W1 and timing."""
    X_tr = np.asarray(data["X_train_dist"], dtype=float)
    C_tr = np.asarray(data["C_train_scalar"], dtype=float)
    Y_tr = np.asarray(data["Y_train"], dtype=float)
    X_te = np.asarray(data["X_test_dist"], dtype=float)
    C_te = np.asarray(data["C_test_scalar"], dtype=float)
    Y_te = np.asarray(data["Y_test"], dtype=float)

    t0 = time.perf_counter()
    model.fit(X_tr, C_tr, Y_tr)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    q_pred = model.predict_quantiles(X_te, C_te, u_grid_eval)
    q_true = _quantiles_from_samples(Y_te, u_grid_eval)
    w1_each = quantile_w1(q_pred, q_true)
    test_time = time.perf_counter() - t1

    return {
        "method": method_name,
        "w1": float(np.mean(w1_each)),
        "fit_time": float(fit_time),
        "test_time": float(test_time),
        "total_train_test_time": float(fit_time + test_time),
    }


def make_methods(
    m_fit: int = 120,
    m_bins: int = 80,
    structured_max_iter: int = 80,
    tw_bandwidth: Optional[float] = None,
    tw_bandwidth_scale: float = 1.0,
    tw_pred_sample_size: Optional[int] = None,
):
    """Construct fresh method objects for one Monte Carlo replicate."""
    return {
        "structured_transport": Fig4StructuredTransportRegressor(
            m_bins=m_bins,
            max_iter=structured_max_iter,
            tol=1e-6,
        ),
        "TW2025_PGFR": ExternalTuckerWuPGFRRegressor(
            n_quantiles=m_fit,
            bandwidth=tw_bandwidth,
            bandwidth_scale=tw_bandwidth_scale,
            quantile_eps=1e-4,
            pred_sample_size=tw_pred_sample_size,
        ),
    }


def run_one_replicate(
    rep: int,
    seed: int,
    N_train: int = 100,
    N_test: int = 1000,
    n_obs: int = 100,
    m_fit: int = 120,
    m_bins: int = 80,
    pis: Tuple[float, float, float] = (0.3, 0.4, 0.3),
    J_segments: int = 8,
    phi0_center: float = 1.0,
    phi0_amplitude: float = 0.3,
    phi1_center: float = 0.0,
    phi1_amplitude: float = 0.3,
    domain_min: float = 0.0,
    domain_max: float = 4.0,
    domain_buffer: float = 0.25,
    structured_max_iter: int = 80,
    tw_bandwidth: Optional[float] = None,
    tw_bandwidth_scale: float = 1.0,
    tw_pred_sample_size: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Generate one shared dataset and evaluate both methods."""
    data = generate_friedman10_data(
        N_train=N_train,
        N_test=N_test,
        n_obs=n_obs,
        pis=pis,
        J_segments=J_segments,
        phi0_center=phi0_center,
        phi0_amplitude=phi0_amplitude,
        phi1_center=phi1_center,
        phi1_amplitude=phi1_amplitude,
        domain_min=domain_min,
        domain_max=domain_max,
        domain_buffer=domain_buffer,
        seed=seed,
    )

    methods = make_methods(
        m_fit=m_fit,
        m_bins=m_bins,
        structured_max_iter=structured_max_iter,
        tw_bandwidth=tw_bandwidth,
        tw_bandwidth_scale=tw_bandwidth_scale,
        tw_pred_sample_size=tw_pred_sample_size,
    )
    u_grid_eval = make_quantile_grid(m_fit)
    run_rows: List[Dict[str, object]] = []

    for method_index, (method_name, model) in enumerate(methods.items()):
        # Reset method-side random states so results remain reproducible even if
        # the method order is later changed or one method fails.
        method_seed = int(seed + 10000 + 1000 * method_index)
        np.random.seed(method_seed)
        torch.manual_seed(method_seed)

        try:
            if verbose:
                print(f"    fitting {method_name} ...", flush=True)
            run_result = fit_predict_score_method(
                method_name=method_name,
                model=model,
                data=data,
                u_grid_eval=u_grid_eval,
            )
            run_result["status"] = "ok"
            run_rows.append(run_result)

            if verbose:
                print(
                    f"      {method_name}: mean W1={run_result['w1']:.4e}, "
                    f"time={run_result['total_train_test_time']:.2f}s",
                    flush=True,
                )
        except Exception as exc:
            run_rows.append(
                {
                    "method": method_name,
                    "w1": np.nan,
                    "fit_time": np.nan,
                    "test_time": np.nan,
                    "total_train_test_time": np.nan,
                    "status": "failed",
                    "error": repr(exc),
                }
            )
            print(
                f"[WARNING] {method_name} failed for rep={rep}, seed={seed}: {exc}",
                flush=True,
            )

    return run_rows, dict(data["config"])


def save_results(results: Dict[str, Dict[str, Dict[str, object]]], output_mat: str) -> Path:
    output_path = Path(output_mat)
    if output_path.suffix.lower() != ".mat":
        output_path = output_path.with_suffix(".mat")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, output_path)
    return output_path


def run_experiment(
    reps: int = 100,
    N_train: int = 100,
    N_test: int = 1000,
    n_obs: int = 100,
    m_fit: int = 120,
    m_bins: int = 80,
    pis: Tuple[float, float, float] = (0.3, 0.4, 0.3),
    J_segments: int = 8,
    phi0_center: float = 1.0,
    phi0_amplitude: float = 0.3,
    phi1_center: float = 0.0,
    phi1_amplitude: float = 0.3,
    domain_min: float = 0.0,
    domain_max: float = 4.0,
    domain_buffer: float = 0.25,
    base_seed: int = 2026,
    structured_max_iter: int = 80,
    tw_bandwidth: Optional[float] = None,
    tw_bandwidth_scale: float = 1.0,
    tw_pred_sample_size: Optional[int] = None,
    output_mat: str = "results/friedman10_structured_transport_pgfr_results.mat",
    verbose: bool = True,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Run all Monte Carlo replicates and save results in the existing .mat style."""
    setting_key = f"{RESULT_KEY}_N_{int(N_train)}"
    results: Dict[str, Dict[str, Dict[str, object]]] = {
        setting_key: {
            method_name: {"w1": [], "time": 0.0}
            for method_name in METHOD_NAMES
        }
    }
    failure_counts = {method_name: 0 for method_name in METHOD_NAMES}
    setting_results = results[setting_key]

    for rep in range(int(reps)):
        seed = int(base_seed + rep)
        if verbose:
            print(f"[rep={rep + 1}/{reps}, seed={seed}]", flush=True)

        run_rows, _data_config = run_one_replicate(
            rep=rep,
            seed=seed,
            N_train=N_train,
            N_test=N_test,
            n_obs=n_obs,
            m_fit=m_fit,
            m_bins=m_bins,
            pis=pis,
            J_segments=J_segments,
            phi0_center=phi0_center,
            phi0_amplitude=phi0_amplitude,
            phi1_center=phi1_center,
            phi1_amplitude=phi1_amplitude,
            domain_min=domain_min,
            domain_max=domain_max,
            domain_buffer=domain_buffer,
            structured_max_iter=structured_max_iter,
            tw_bandwidth=tw_bandwidth,
            tw_bandwidth_scale=tw_bandwidth_scale,
            tw_pred_sample_size=tw_pred_sample_size,
            verbose=verbose,
        )

        for row in run_rows:
            method_name = str(row["method"])
            if row["status"] == "ok":
                method_result = setting_results.setdefault(method_name, {"w1": [], "time": 0.0})
                method_result["w1"].append(float(row["w1"]))
                method_result["time"] += float(row["total_train_test_time"])
            else:
                failure_counts[method_name] = failure_counts.get(method_name, 0) + 1

        save_results(results, output_mat)

    output_path = save_results(results, output_mat)

    if verbose:
        print(f"\nSaved results to: {output_path.resolve()}")
        print(f"\nSummary for {setting_key}:")
        for method_name in METHOD_NAMES:
            method_result = setting_results.get(method_name, {})
            w1_values = np.asarray(method_result.get("w1", []), dtype=float)
            if w1_values.size > 0:
                sd = float(np.std(w1_values, ddof=1)) if w1_values.size > 1 else 0.0
                print(
                    f"  {method_name}: "
                    f"w1_len={w1_values.size}, mean={float(np.mean(w1_values)):.6g}, "
                    f"sd={sd:.6g}, time={float(method_result.get('time', 0.0)):.6g}s, "
                    f"failed={failure_counts.get(method_name, 0)}",
                    flush=True,
                )
            else:
                print(
                    f"  {method_name}: no successful runs, "
                    f"failed={failure_counts.get(method_name, 0)}",
                    flush=True,
                )

    return results


# ============================================================
# Command-line interface
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare structured transport and PGFR under a 10-scalar "
            "Friedman-type nonlinear data-generating mechanism."
        )
    )
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-obs", type=int, default=100)
    parser.add_argument("--m-fit", type=int, default=120)
    parser.add_argument("--m-bins", type=int, default=80)

    parser.add_argument("--pis", type=float, nargs=3, default=[0.3, 0.4, 0.3])
    parser.add_argument("--J-segments", type=int, default=8)

    parser.add_argument("--phi0-center", type=float, default=1.0)
    parser.add_argument("--phi0-amplitude", type=float, default=0.3)
    parser.add_argument("--phi1-center", type=float, default=0.0)
    parser.add_argument("--phi1-amplitude", type=float, default=0.3)

    parser.add_argument("--domain-min", type=float, default=0.0)
    parser.add_argument("--domain-max", type=float, default=4.0)
    parser.add_argument("--domain-buffer", type=float, default=0.25)
    parser.add_argument("--base-seed", type=int, default=2026)

    parser.add_argument("--structured-max-iter", type=int, default=80)
    parser.add_argument("--tw-bandwidth", type=float, default=None)
    parser.add_argument("--tw-bandwidth-scale", type=float, default=1.0)
    parser.add_argument(
        "--tw-pred-sample-size",
        type=int,
        default=None,
        help=(
            "Pseudo-sample size used by the external PGFR predictor at each "
            "test point. If omitted, n_obs is used."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results_nonlinear/friedman10_structured_transport_pgfr_results.mat",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    return run_experiment(
        reps=args.reps,
        N_train=args.n_train,
        N_test=args.n_test,
        n_obs=args.n_obs,
        m_fit=args.m_fit,
        m_bins=args.m_bins,
        pis=(float(args.pis[0]), float(args.pis[1]), float(args.pis[2])),
        J_segments=args.J_segments,
        phi0_center=args.phi0_center,
        phi0_amplitude=args.phi0_amplitude,
        phi1_center=args.phi1_center,
        phi1_amplitude=args.phi1_amplitude,
        domain_min=args.domain_min,
        domain_max=args.domain_max,
        domain_buffer=args.domain_buffer,
        base_seed=args.base_seed,
        structured_max_iter=args.structured_max_iter,
        tw_bandwidth=args.tw_bandwidth,
        tw_bandwidth_scale=args.tw_bandwidth_scale,
        tw_pred_sample_size=args.tw_pred_sample_size,
        output_mat=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    main()
