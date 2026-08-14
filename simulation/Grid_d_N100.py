#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
VERSION: fig3_vector_data_p_grid_decay_theta_no_intercept_v3

Compare four methods under a fig3.py / Structured_transport_map_core.py style simulation, but with
one distributional predictor and d=p-1 scalar predictors.

Data setting
------------
For each observation i:

    X_i ~ mu_i, where mu_i is a random mixture of three Beta distributions;
    C_i in R^{p-1} is a nonnegative scalar-vector predictor;
    Y_i = S_eps( S_0(X_i; C_i) ).

The structural regression map is

    S_0(x; C) = phi0(C) * zeta_4(x) + phi1(C),
    phi0(C) = b0 + a0^T C,
    phi1(C) = b1 + a1^T C.

The true coefficient vectors theta0=(b0,a0) and theta1=(b1,a1) are fixed
deterministically for each p.  In the default no-intercept design,

    b0 = b1 = 0,
    a0_j is proportional to j^{-0.8},
    a1_j is proportional to j^{-0.5},

and the two coefficient vectors are normalized so that
||a0||_2^2=phi0_l2_sq and ||a1||_2^2=phi1_l2_sq.  With positive scalar
covariates, phi0(C)=a0^T C stays positive without relying on an intercept.
The true data-generating model is fixed across Monte Carlo replicates for a
given p, and the seed controls both the local RNG and the global np.random
calls used inside Structured_transport_map_core.py.

Methods
-------
1. GOT_noisy_scalar
   Uses GOT1DMixedRegressor.  Only this method converts scalar predictors into
   Gaussian-smoothed pseudo-distributions.

2. structured_transport
   Directly calls fig4.prepare_grid_data and fig4.fit_ours_cvec_from_grid.

3. GP2022_condfree_OT
   Strict distribution-on-distribution baseline.  It ignores all scalar
   predictors and directly calls fig4.fit_condfree_from_grid.

4. TW2025_PGFR
   Calls partially_global_frechet.fit_pgfr_baseline_from_samples and predicts
   through the external PGFR predictor at each test point.

Outputs
-------
The script writes a MATLAB .mat file containing one variable named ``results``.
Its nested structure is

    results[p_key][method] = {
        "w1": [mean W1 from each successful run],
        "time": total training-and-testing time across successful runs,
    }

where ``p_key`` is ``p_2``, ``p_5``, etc. because MATLAB structure-field names
cannot start with a digit.

No Boolean command-line switches are used.  All command-line arguments are
value-based.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


# ============================================================
# Robust local imports
# ============================================================

def _load_phi_module():
    """Load Structured_transport_map_core.py robustly, including uploaded names like phi_0_1(2).py."""
    try:
        import Structured_transport_map_core  # type: ignore
        return Structured_transport_map_core
    except Exception:
        pass

    candidates = [
        _THIS_DIR / "Structured_transport_map_core.py",
        _THIS_DIR / "phi_0_1(2).py",
        _THIS_DIR / "phi_0_1(1).py",
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

    raise ImportError("Could not locate Structured_transport_map_core.py or phi_0_1(...).py in the script folder.")


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
fig4_fit_condfree_from_grid = fig4_mod.fit_condfree_from_grid
fig4_fit_ours_cvec_from_grid = fig4_mod.fit_ours_cvec_from_grid

try:
    from got import (
        GOT1DMixedRegressor,
        make_quantile_grid,
        samples_to_quantiles as got_samples_to_quantiles,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Could not import got.py. "
        "Please put this script in the same folder as that file."
    ) from exc

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
    """Convert arrays/lists and NumPy scalars into serializable values."""
    if x is None:
        return ""
    if isinstance(x, np.ndarray):
        return json.dumps(x.tolist())
    if isinstance(x, (list, tuple)):
        return json.dumps(list(x))
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def _quantiles_from_samples(X: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    return got_samples_to_quantiles(np.asarray(X, dtype=float), u_grid)


def quantile_w1(q_pred: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """Approximate 1D W1 by averaging absolute quantile differences."""
    q_pred = np.asarray(q_pred, dtype=float)
    q_true = np.asarray(q_true, dtype=float)
    if q_pred.shape != q_true.shape:
        raise ValueError(f"q_pred and q_true shapes differ: {q_pred.shape} vs {q_true.shape}")
    return np.mean(np.abs(q_pred - q_true), axis=-1)


def _resample_distribution_samples(X: np.ndarray, n_required: int) -> np.ndarray:
    """Represent each empirical distribution by n_required deterministic quantile samples."""
    X = np.asarray(X, dtype=float)
    if X.shape[1] == int(n_required):
        return X
    u_tmp = (np.arange(int(n_required), dtype=float) + 0.5) / int(n_required)
    return np.vstack([np.quantile(row, u_tmp) for row in X])


# ============================================================
# fig3.py-style vector-covariate data generation from Structured_transport_map_core.py
# ============================================================

def make_decay_no_intercept_coefficients(
    p: int,
    phi0_l2_sq: float = 1.0,
    phi1_l2_sq: float = 0.25,
    decay0: float = 0.8,
    decay1: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic nonuniform theta0 and theta1 with zero intercepts.

    Here p is the total predictor count, so the scalar dimension is d=p-1.
    The coefficient vectors theta0=(b0,a0) and theta1=(b1,a1) have length p,
    where theta[0] is the intercept.  We set

        b0 = b1 = 0,
        a0_j proportional to j^{-decay0},
        a1_j proportional to j^{-decay1},      j=1,...,d,

    and normalize so that ||a0||_2^2=phi0_l2_sq and
    ||a1||_2^2=phi1_l2_sq.  With scalar covariates sampled from a positive
    interval, phi0(C)=a0^T C is positive without using an intercept.
    """
    p = int(p)
    if p < 2:
        raise ValueError("This no-intercept design requires p >= 2, i.e. at least one scalar predictor.")
    if phi0_l2_sq <= 0 or phi1_l2_sq < 0:
        raise ValueError("Require phi0_l2_sq > 0 and phi1_l2_sq >= 0.")

    d = p - 1
    j = np.arange(1, d + 1, dtype=float)

    raw0 = j ** (-float(decay0))
    raw1 = j ** (-float(decay1))

    a0 = np.sqrt(float(phi0_l2_sq)) * raw0 / max(float(np.linalg.norm(raw0)), 1e-12)
    if float(phi1_l2_sq) == 0:
        a1 = np.zeros_like(raw1)
    else:
        a1 = np.sqrt(float(phi1_l2_sq)) * raw1 / max(float(np.linalg.norm(raw1)), 1e-12)

    theta0 = np.concatenate([[0.0], a0])
    theta1 = np.concatenate([[0.0], a1])
    return theta0, theta1

def evaluate_phi(C: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate theta[0] + C @ theta[1:]."""
    C = _as_2d_c(C)
    theta = np.asarray(theta, dtype=float)
    if theta.shape[0] != C.shape[1] + 1:
        raise ValueError("theta length must equal C dimension + 1.")
    return theta[0] + C @ theta[1:]


def S0_true_vector(x: np.ndarray, C: np.ndarray, theta0: np.ndarray, theta1: np.ndarray) -> np.ndarray:
    """True monotone map S0(x;C)=phi0(C) zeta_4(x)+phi1(C)."""
    x = np.asarray(x, dtype=float)
    c = np.asarray(C, dtype=float).reshape(-1)
    phi0 = float(theta0[0] + np.dot(theta0[1:], c))
    phi1 = float(theta1[0] + np.dot(theta1[1:], c))
    return phi0 * phi.zeta_k(x, k=4) + phi1


def _scaled_scalar_matrix(
    rng: np.random.Generator,
    N: int,
    d: int,
    scalar_range: Tuple[float, float],
    scalar_scaling: str = "none",
) -> np.ndarray:
    """Generate nonnegative scalar covariates.

    The default is no scaling.  The true coefficients have fixed L2 norms and
    nonuniform polynomial decay; scalar_range is positive by default so that
    phi0(C)=a0^T C remains safely positive even with zero intercept.
    """
    if d <= 0:
        return np.zeros((int(N), 0), dtype=float)
    lo, hi = map(float, scalar_range)
    if lo < 0:
        raise ValueError("For this data generator scalar_range must be nonnegative, e.g. 0 1.")
    C = rng.uniform(lo, hi, size=(int(N), int(d)))
    if scalar_scaling == "sqrt_d":
        C = C / np.sqrt(float(d))
    elif scalar_scaling == "none":
        pass
    else:
        raise ValueError("scalar_scaling must be either 'sqrt_d' or 'none'.")
    return C


def _theoretical_s0_upper(
    d: int,
    scalar_range: Tuple[float, float],
    scalar_scaling: str,
    theta0: np.ndarray,
    theta1: np.ndarray,
) -> float:
    """Conservative upper bound of S0 over x in [0,1] and C in the covariate box."""
    if d <= 0:
        cmax_vec = np.zeros(0, dtype=float)
    else:
        cmax = float(scalar_range[1])
        if scalar_scaling == "sqrt_d":
            cmax = cmax / np.sqrt(float(d))
        cmax_vec = np.full(int(d), cmax, dtype=float)
    phi0_max = float(theta0[0] + np.dot(theta0[1:], cmax_vec))
    phi1_max = float(theta1[0] + np.dot(theta1[1:], cmax_vec))
    # zeta_4 maps [0,1] to [0,1].
    return phi0_max + phi1_max


def generate_fig3_vector_style_data(
    N_train: int = 100,
    N_test: int = 100,
    p: int = 5,
    n_obs: int = 100,
    pis: Tuple[float, float, float] = (0.3, 0.4, 0.3),
    J_segments: int = 8,
    scalar_range: Tuple[float, float] = (0.5, 1.5),
    scalar_scaling: str = "none",
    phi0_l2_sq: float = 1.0,
    phi1_l2_sq: float = 0.25,
    theta_decay0: float = 0.8,
    theta_decay1: float = 0.5,
    domain_max: float = 4.0,
    domain_buffer: float = 0.25,
    seed: int = 2026,
) -> Dict[str, np.ndarray]:
    """Generate fig3-style data with vector-valued scalar covariates.

    Only the data-generating map is changed relative to the mixed-GOT script;
    the returned dictionary keeps the same keys used by the method wrappers.
    """
    # Important: Structured_transport_map_core.py uses the global np.random API internally.
    # Setting both seeds makes each replicate exactly reproducible.
    np.random.seed(int(seed))
    rng = np.random.default_rng(int(seed))

    p = int(p)
    if p < 2:
        raise ValueError("This no-intercept theta design requires p >= 2, i.e. at least one scalar predictor.")
    d = p - 1

    # Fixed true coefficients for all Monte Carlo replicates at the same p.
    # theta0 and theta1 include the intercept, so their length is p=d+1.
    # In this design b0=b1=0 and the scalar coefficients decay polynomially.
    theta0, theta1 = make_decay_no_intercept_coefficients(
        p=p,
        phi0_l2_sq=phi0_l2_sq,
        phi1_l2_sq=phi1_l2_sq,
        decay0=theta_decay0,
        decay1=theta_decay1,
    )

    s0_upper = _theoretical_s0_upper(
        d=d,
        scalar_range=scalar_range,
        scalar_scaling=scalar_scaling,
        theta0=theta0,
        theta1=theta1,
    )
    domain_max_used = max(float(domain_max), float(s0_upper + domain_buffer))

    def _make_split(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.empty((int(N), int(n_obs)), dtype=float)
        Y = np.empty((int(N), int(n_obs)), dtype=float)
        C = _scaled_scalar_matrix(
            rng=rng,
            N=int(N),
            d=d,
            scalar_range=scalar_range,
            scalar_scaling=scalar_scaling,
        )

        for i in range(int(N)):
            # Use Structured_transport_map_core.py's mixture-Beta generation primitives.
            alphas, betas = phi.sample_mu_params()
            Xi = phi.sample_from_mixture_beta(alphas, betas, pis, size=int(n_obs))
            S_eps = phi.sample_random_S_eps(
                J=int(J_segments),
                domain_min=0.0,
                domain_max=float(domain_max_used),
            )
            S0_vals = S0_true_vector(Xi, C[i], theta0=theta0, theta1=theta1)
            Yi = S_eps(S0_vals)

            X[i, :] = np.asarray(Xi, dtype=float)
            Y[i, :] = np.asarray(Yi, dtype=float)

        return X, Y, C

    X_train, Y_train, C_train = _make_split(int(N_train))
    X_test, Y_test, C_test = _make_split(int(N_test))

    return {
        "X_train_dist": X_train,
        "Y_train": Y_train,
        "C_train_scalar": C_train,
        "X_test_dist": X_test,
        "Y_test": Y_test,
        "C_test_scalar": C_test,
        "theta0_true": theta0,
        "theta1_true": theta1,
        "config": {
            "data_source": "fig3_phi_0_1_vector_covariates",
            "N_train": int(N_train),
            "N_test": int(N_test),
            "p": int(p),
            "n_scalar": int(d),
            "n_obs": int(n_obs),
            "pis": list(pis),
            "J_segments": int(J_segments),
            "scalar_range_raw": list(scalar_range),
            "scalar_scaling": scalar_scaling,
            "theta_rule": "decay_no_intercept",
            "phi0_l2_sq": float(phi0_l2_sq),
            "phi1_l2_sq": float(phi1_l2_sq),
            "theta_decay0": float(theta_decay0),
            "theta_decay1": float(theta_decay1),
            "domain_max_requested": float(domain_max),
            "domain_max_used": float(domain_max_used),
            "s0_upper_bound": float(s0_upper),
            "seed": int(seed),
        },
    }


# ============================================================
# Method 1: GOT with scalar-to-noisy-distribution conversion
# ============================================================

@dataclass
class GOTNoisyScalarMethod:
    m: int = 120
    scalar_noise_sd: float = 0.03
    deterministic_scalar_noise: bool = True
    scalar_seed: int = 12345
    fit_order: bool = True
    alpha_bounds: Tuple[float, float] = (0.0, 1.0)
    order_maxiter: int = 40
    final_maxiter: int = 100
    verbose: bool = False

    model_: Optional[GOT1DMixedRegressor] = field(default=None, init=False)

    def fit(self, X_dist: np.ndarray, C: np.ndarray, Y: np.ndarray):
        self.model_ = GOT1DMixedRegressor(
            m=self.m,
            scalar_noise_sd=self.scalar_noise_sd,
            deterministic_scalar_noise=self.deterministic_scalar_noise,
            scalar_seed=self.scalar_seed,
            fit_order=self.fit_order,
            order=None,
            alpha_bounds=self.alpha_bounds,
            order_maxiter=self.order_maxiter,
            final_maxiter=self.final_maxiter,
            verbose=self.verbose,
        )
        self.model_.fit(X_dist, C, Y)
        return self

    def predict_quantiles(self, X_dist: np.ndarray, C: np.ndarray, u_grid_eval: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")

        X_dist = np.asarray(X_dist, dtype=float)
        n_required = int(self.model_.n_obs_ or X_dist.shape[1])
        X_use = _resample_distribution_samples(X_dist, n_required=n_required)

        q_pred = self.model_.predict_quantiles(X_use, C)
        u_model = self.model_.u_grid_
        if u_model is None:
            raise RuntimeError("GOT model has no quantile grid.")
        if len(u_model) != len(u_grid_eval) or np.max(np.abs(u_model - u_grid_eval)) > 1e-12:
            q_new = np.empty((q_pred.shape[0], len(u_grid_eval)), dtype=float)
            for i in range(q_pred.shape[0]):
                q_new[i] = np.interp(u_grid_eval, u_model, q_pred[i])
            q_pred = q_new
        return q_pred

    def info(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "scalar_noise_sd": self.scalar_noise_sd,
            "fit_order": self.fit_order,
            "source": "got_1d_mixed_distribution_scalar_predictors.GOT1DMixedRegressor",
        }
        if self.model_ is not None:
            out["training_loss"] = float(self.model_.fit_info_.get("training_loss", np.nan))
            out["order_hat"] = _safe_jsonable(self.model_.order_)
            out["alpha_hat"] = _safe_jsonable(self.model_.alpha_)
        return out


# ============================================================
# Method 2: Our structured transport, directly calling base_utils.py
# ============================================================

@dataclass
class Fig4StructuredTransportRegressor:
    """Structured transport map using base_utils.py's Section 4.2 implementation."""
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

    def predict_quantiles(self, X_dist: np.ndarray, C: np.ndarray, u_grid_eval: np.ndarray) -> np.ndarray:
        if self.S_hat_func_ is None or self.n_scalar_ is None:
            raise RuntimeError("Model not fitted.")

        X_dist = np.asarray(X_dist, dtype=float)
        C = _as_2d_c(C, n=X_dist.shape[0])
        if C.shape[1] != self.n_scalar_:
            raise ValueError(f"Expected {self.n_scalar_} scalar covariates, got {C.shape[1]}.")

        qX = _quantiles_from_samples(X_dist, u_grid_eval)
        q_pred = np.empty_like(qX, dtype=float)
        for i in range(qX.shape[0]):
            q = self.S_hat_func_(qX[i], C[i])
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
# Method 3: GP2022 baseline, ignoring scalars, directly calling base_utils.py
# ============================================================

@dataclass
class Fig4GP2022IgnoreScalarRegressor:
    """Strict distribution-on-distribution OT-map baseline."""
    m_bins: int = 80
    enforce_monotone_prediction: bool = True

    centers_: Optional[np.ndarray] = field(default=None, init=False)
    z_: Optional[np.ndarray] = field(default=None, init=False)
    S_hat_func_: Optional[object] = field(default=None, init=False)

    def fit(self, X_dist: np.ndarray, C: np.ndarray, Y: np.ndarray):
        X_list = [np.asarray(x, dtype=float) for x in np.asarray(X_dist)]
        Y_list = [np.asarray(y, dtype=float) for y in np.asarray(Y)]

        centers, w_ij, y_ij = fig4_prepare_grid_data(
            X_list=X_list,
            Y_list=Y_list,
            m_bins=self.m_bins,
        )
        z_hat, S_hat = fig4_fit_condfree_from_grid(centers, w_ij, y_ij)

        self.centers_ = np.asarray(centers, dtype=float)
        self.z_ = np.asarray(z_hat, dtype=float)
        self.S_hat_func_ = S_hat
        return self

    def predict_quantiles(self, X_dist: np.ndarray, C: np.ndarray, u_grid_eval: np.ndarray) -> np.ndarray:
        if self.S_hat_func_ is None:
            raise RuntimeError("Model not fitted.")
        qX = _quantiles_from_samples(np.asarray(X_dist, dtype=float), u_grid_eval)
        q_pred = np.empty_like(qX, dtype=float)
        for i in range(qX.shape[0]):
            q = self.S_hat_func_(qX[i])
            if self.enforce_monotone_prediction:
                q = _ensure_increasing(q)
            q_pred[i] = q
        return q_pred

    def info(self) -> Dict[str, object]:
        return {
            "m_bins": self.m_bins,
            "use_scalar": False,
            "source": "fig4.fit_condfree_from_grid",
        }


# ============================================================
# Method 4: TW2025 PGFR, calling partially_global_frechet.py
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

    def predict_quantiles(self, X_dist: np.ndarray, C: np.ndarray, u_grid_eval: np.ndarray) -> np.ndarray:
        if self.predictor_ is None or self.info_ is None:
            raise RuntimeError("External PGFR predictor not fitted.")

        X_dist = np.asarray(X_dist, dtype=float)
        C = _as_2d_c(C, n=X_dist.shape[0])
        n_pred = int(self.pred_sample_size or X_dist.shape[1])

        q_out = np.empty((X_dist.shape[0], len(u_grid_eval)), dtype=float)
        for i in range(X_dist.shape[0]):
            y_pred_sample = self.predictor_(X_dist[i], C[i], n_pred=n_pred)
            q_out[i] = np.quantile(np.asarray(y_pred_sample, dtype=float), u_grid_eval)
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
    data: Dict[str, np.ndarray],
    u_grid_eval: np.ndarray,
) -> Dict[str, object]:
    """Fit one method, predict, and return its run-level W1 and timing."""
    X_tr = data["X_train_dist"]
    C_tr = data["C_train_scalar"]
    Y_tr = data["Y_train"]
    X_te = data["X_test_dist"]
    C_te = data["C_test_scalar"]
    Y_te = data["Y_test"]

    t0 = time.perf_counter()
    model.fit(X_tr, C_tr, Y_tr)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    q_pred = model.predict_quantiles(X_te, C_te, u_grid_eval)
    q_true = _quantiles_from_samples(Y_te, u_grid_eval)
    w1_each = quantile_w1(q_pred, q_true)
    test_time = time.perf_counter() - t1

    run_result: Dict[str, object] = {
        "method": method_name,
        "w1": float(np.mean(w1_each)),
        "total_train_test_time": float(fit_time + test_time),
    }
    return run_result


def make_methods(
    p: int,
    seed: int,
    m_fit: int = 120,
    m_bins: int = 80,
    scalar_noise_sd: float = 0.03,
    got_order_maxiter: int = 40,
    got_final_maxiter: int = 100,
    tw_bandwidth: Optional[float] = None,
    tw_bandwidth_scale: float = 1.0,
    tw_pred_sample_size: Optional[int] = None,
    verbose: bool = False,
):
    """Construct fresh method objects for one replicate under the standard setting."""
    return {
        "GOT_noisy_scalar": GOTNoisyScalarMethod(
            m=m_fit,
            scalar_noise_sd=scalar_noise_sd,
            deterministic_scalar_noise=True,
            scalar_seed=seed + 777,
            fit_order=True,
            order_maxiter=got_order_maxiter,
            final_maxiter=got_final_maxiter,
            verbose=verbose,
        ),
        "structured_transport": Fig4StructuredTransportRegressor(
            m_bins=m_bins,
            max_iter=80,
            tol=1e-6,
        ),
        "GP2022_condfree_OT": Fig4GP2022IgnoreScalarRegressor(
            m_bins=m_bins,
        ),
        "TW2025_PGFR": ExternalTuckerWuPGFRRegressor(
            n_quantiles=m_fit,
            bandwidth=tw_bandwidth,
            bandwidth_scale=tw_bandwidth_scale,
            quantile_eps=1e-4,
            pred_sample_size=tw_pred_sample_size,
        ),
    }


def run_one_replicate_all_methods(
    p: int,
    rep: int,
    seed: int,
    N_train: int = 100,
    N_test: int = 100,
    n_obs: int = 100,
    m_fit: int = 120,
    m_bins: int = 80,
    pis: Tuple[float, float, float] = (0.3, 0.4, 0.3),
    J_segments: int = 8,
    scalar_range: Tuple[float, float] = (0.5, 1.5),
    scalar_scaling: str = "none",
    phi0_l2_sq: float = 1.0,
    phi1_l2_sq: float = 0.25,
    theta_decay0: float = 0.8,
    theta_decay1: float = 0.5,
    domain_max: float = 4.0,
    domain_buffer: float = 0.25,
    scalar_noise_sd: float = 0.03,
    got_order_maxiter: int = 40,
    got_final_maxiter: int = 100,
    tw_bandwidth: Optional[float] = None,
    tw_bandwidth_scale: float = 1.0,
    tw_pred_sample_size: Optional[int] = None,
    verbose: bool = False,
) -> List[Dict[str, object]]:
    """Generate one fig3-vector dataset and evaluate all methods on the same split."""
    data = generate_fig3_vector_style_data(
        N_train=N_train,
        N_test=N_test,
        p=p,
        n_obs=n_obs,
        pis=pis,
        J_segments=J_segments,
        scalar_range=scalar_range,
        scalar_scaling=scalar_scaling,
        phi0_l2_sq=phi0_l2_sq,
        phi1_l2_sq=phi1_l2_sq,
        theta_decay0=theta_decay0,
        theta_decay1=theta_decay1,
        domain_max=domain_max,
        domain_buffer=domain_buffer,
        seed=seed,
    )

    methods = make_methods(
        p=p,
        seed=seed,
        m_fit=m_fit,
        m_bins=m_bins,
        scalar_noise_sd=scalar_noise_sd,
        got_order_maxiter=got_order_maxiter,
        got_final_maxiter=got_final_maxiter,
        tw_bandwidth=tw_bandwidth,
        tw_bandwidth_scale=tw_bandwidth_scale,
        tw_pred_sample_size=tw_pred_sample_size,
        verbose=verbose,
    )

    u_grid_eval = make_quantile_grid(m_fit)
    run_rows: List[Dict[str, object]] = []

    for method_name, model in methods.items():
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
            run_rows.append({
                "method": method_name,
                "w1": np.nan,
                "total_train_test_time": np.nan,
                "status": "failed",
            })
            print(f"[WARNING] {method_name} failed for p={p}, rep={rep}: {exc}", flush=True)

    return run_rows


def run_p_grid_experiment(
    p_list: Sequence[int] = (2, 5, 10, 20),
    reps: int = 20,
    N_train: int = 100,
    N_test: int = 100,
    n_obs: int = 100,
    m_fit: int = 120,
    m_bins: int = 80,
    pis: Tuple[float, float, float] = (0.3, 0.4, 0.3),
    J_segments: int = 8,
    scalar_range: Tuple[float, float] = (0.5, 1.5),
    scalar_scaling: str = "none",
    phi0_l2_sq: float = 1.0,
    phi1_l2_sq: float = 0.25,
    theta_decay0: float = 0.8,
    theta_decay1: float = 0.5,
    domain_max: float = 4.0,
    domain_buffer: float = 0.25,
    scalar_noise_sd: float = 0.03,
    base_seed: int = 2026,
    got_order_maxiter: int = 40,
    got_final_maxiter: int = 100,
    tw_bandwidth: Optional[float] = None,
    tw_bandwidth_scale: float = 1.0,
    tw_pred_sample_size: Optional[int] = None,
    output_mat: str = "results/fig3_vector_decay_theta_no_intercept_results.mat",
    verbose: bool = True,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Run the full p-grid experiment and save W1 lists and total times."""
    results: Dict[str, Dict[str, Dict[str, object]]] = {}

    for p in p_list:
        p_key = f"p_{int(p)}"
        p_results = results.setdefault(p_key, {})

        for rep in range(int(reps)):
            seed = int(base_seed + 100000 * int(p) + rep)
            if verbose:
                print(f"[p={p}, rep={rep + 1}/{reps}, seed={seed}]", flush=True)

            run_rows = run_one_replicate_all_methods(
                p=int(p),
                rep=int(rep),
                seed=seed,
                N_train=N_train,
                N_test=N_test,
                n_obs=n_obs,
                m_fit=m_fit,
                m_bins=m_bins,
                pis=pis,
                J_segments=J_segments,
                scalar_range=scalar_range,
                scalar_scaling=scalar_scaling,
                phi0_l2_sq=phi0_l2_sq,
                phi1_l2_sq=phi1_l2_sq,
                theta_decay0=theta_decay0,
                theta_decay1=theta_decay1,
                domain_max=domain_max,
                domain_buffer=domain_buffer,
                scalar_noise_sd=scalar_noise_sd,
                got_order_maxiter=got_order_maxiter,
                got_final_maxiter=got_final_maxiter,
                tw_bandwidth=tw_bandwidth,
                tw_bandwidth_scale=tw_bandwidth_scale,
                tw_pred_sample_size=tw_pred_sample_size,
                verbose=verbose,
            )

            for row in run_rows:
                method_name = str(row["method"])
                method_result = p_results.setdefault(
                    method_name,
                    {"w1": [], "time": 0.0},
                )
                if row["status"] == "ok":
                    method_result["w1"].append(float(row["w1"]))
                    method_result["time"] += float(row["total_train_test_time"])

    output_path = Path(output_mat)
    if output_path.suffix.lower() != ".mat":
        output_path = output_path.with_suffix(".mat")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results,output_path)

    if verbose:
        print(f"\nSaved MATLAB results to: {output_path.resolve()}")

    return results


# ============================================================
# Command-line interface
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare mixed-predictor methods under fig3-style vector-covariate data."
    )
    parser.add_argument("--p-list", type=int, nargs="+", default=[2, 11],
                        help="Total predictor counts p. There is 1 distribution predictor and p-1 scalars.")
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-obs", type=int, default=100)
    parser.add_argument("--m-fit", type=int, default=120)
    parser.add_argument("--m-bins", type=int, default=80)

    parser.add_argument("--pis", type=float, nargs=3, default=[0.3, 0.4, 0.3])
    parser.add_argument("--J-segments", type=int, default=8)
    parser.add_argument("--scalar-range", type=float, nargs=2, default=[0.0, 1.5],
                        help="Positive scalar range. This no-intercept theta design needs C away from zero.")
    parser.add_argument("--scalar-scaling", type=str, default="none", choices=["none", "sqrt_d"],
                        help="Default none. Optional sqrt_d is retained only for sensitivity checks.")
    parser.add_argument("--phi0-l2-sq", type=float, default=1.0,
                        help="Squared L2 norm of the true scale coefficients a0; intercept b0 is fixed to 0.")
    parser.add_argument("--phi1-l2-sq", type=float, default=0.25,
                        help="Squared L2 norm of the true location coefficients a1; intercept b1 is fixed to 0.")
    parser.add_argument("--theta-decay0", type=float, default=2.0,
                        help="Polynomial decay exponent for a0_j proportional to j^{-theta_decay0}.")
    parser.add_argument("--theta-decay1", type=float, default=1.0,
                        help="Polynomial decay exponent for a1_j proportional to j^{-theta_decay1}.")

    parser.add_argument("--domain-max", type=float, default=4.0,
                        help="Requested upper domain for random S_eps; actual value is enlarged if needed.")
    parser.add_argument("--domain-buffer", type=float, default=0.25)
    parser.add_argument("--scalar-noise-sd", type=float, default=0.03,
                        help="Only used by GOT_noisy_scalar when scalars are converted to pseudo-distributions.")
    parser.add_argument("--base-seed", type=int, default=2026)

    # Standard GOT setting: greedy order search is always used.
    parser.add_argument("--got-order-maxiter", type=int, default=100)
    parser.add_argument("--got-final-maxiter", type=int, default=100)

    # Standard TW2025 setting: use the external PGFR predictor.
    parser.add_argument("--tw-bandwidth", type=float, default=None)
    parser.add_argument("--tw-bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--tw-pred-sample-size", type=int, default=None,
                        help="Pseudo-sample size used by the external PGFR predictor at each test point. "
                             "If omitted, use n_obs. For timing comparable to older fig3.py W2 experiments, try 1000.")

    parser.add_argument("--output", type=str, default="results/fig3_vector_decay_theta_no_intercept_results.mat")
    return parser.parse_args()


def main():
    args = parse_args()

    return run_p_grid_experiment(
        p_list=args.p_list,
        reps=args.reps,
        N_train=args.n_train,
        N_test=args.n_test,
        n_obs=args.n_obs,
        m_fit=args.m_fit,
        m_bins=args.m_bins,
        pis=(float(args.pis[0]), float(args.pis[1]), float(args.pis[2])),
        J_segments=args.J_segments,
        scalar_range=(float(args.scalar_range[0]), float(args.scalar_range[1])),
        scalar_scaling=args.scalar_scaling,
        phi0_l2_sq=args.phi0_l2_sq,
        phi1_l2_sq=args.phi1_l2_sq,
        theta_decay0=args.theta_decay0,
        theta_decay1=args.theta_decay1,
        domain_max=args.domain_max,
        domain_buffer=args.domain_buffer,
        scalar_noise_sd=args.scalar_noise_sd,
        base_seed=args.base_seed,
        got_order_maxiter=args.got_order_maxiter,
        got_final_maxiter=args.got_final_maxiter,
        tw_bandwidth=args.tw_bandwidth,
        tw_bandwidth_scale=args.tw_bandwidth_scale,
        tw_pred_sample_size=args.tw_pred_sample_size,
        output_mat=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    main()
