#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
VERSION: no_flags_fig4_v2_fixed_alpha_seed, direct fig4.py calls with fixed deterministic alpha and seeded simulation.

Compare four methods under the mixed-predictor simulation setting

    predictor 0: one 1D distribution
    predictors 1,...,p-1: scalar predictors

The data generator and GOT baseline are taken from
`got_1d_base.py`, with the true alpha vector
changed to a deterministic polynomial-decay vector with fixed L2 norm.

Important implementation choices
--------------------------------
1. GOT_noisy_scalar
   Uses `GOT1DMixedRegressor`; only this method converts scalar predictors into
   Gaussian-smoothed pseudo-distributions.

2. structured_transport
   Directly calls the Section 4.2 helper functions in `fig4.py`: first
   `prepare_grid_data`, then `fit_ours_cvec_from_grid`.  This avoids duplicating
   the fitting code and keeps the timing consistent with the figure script.

3. GP2022_condfree_OT
   Strict distribution-on-distribution baseline: it ignores all scalar
   predictors.  It directly calls `prepare_grid_data` and
   `fit_condfree_from_grid` from `fig4.py`, so the baseline is fitted by
   the genuine covariate-free routine used in fig4.py.

4. TW2025_PGFR
   Calls `fit_pgfr_baseline_from_samples` / `PartiallyGlobalFrechet1D` from
   `partially_global_frechet.py`.  Its prediction uses the fitted PGFR model,
   hence each test point computes the partially-global weights and solves the
   1D weighted Frechet mean problem with the PAVA projection implemented there.

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
# Imports from the mixed-GOT simulation/baseline file
# ============================================================
try:
    from got_1d_base import (
        GOT1DMixedRegressor,
        generate_mixed_got_1d_data,
        make_quantile_grid,
        samples_to_quantiles as got_samples_to_quantiles,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Could not import got_1d_base.py. "
        "Please put this script in the same folder as that file."
    ) from exc


# ============================================================
# Imports from phi_0_1.py, fig4.py, and partially_global_frechet.py
# ============================================================

def _load_phi_module():
    """Load phi_0_1.py robustly, including uploaded names like phi_0_1(2).py."""
    try:
        import Structured_transport_map_core  # type: ignore
        return Structured_transport_map_core
    except Exception:
        pass

    candidates = [
        _THIS_DIR / "Structured_transport_map_core.py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("phi_0_1_loaded", str(path))
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            # fig4.py imports `phi_0_1` by module name.  When the uploaded file
            # is named `phi_0_1(2).py`, register the loaded module under the
            # canonical name so fig4.py can import it successfully.
            sys.modules.setdefault("phi_0_1", module)
            return module

    raise ImportError("Could not locate phi_0_1.py or phi_0_1(...).py in the script folder.")


phi_mod = _load_phi_module()
# Register explicitly in case phi_0_1.py was loaded from an uploaded name such as
# phi_0_1(2).py.  This is needed before loading fig4.py.
sys.modules.setdefault("phi_0_1", phi_mod)


def _load_fig4_module():
    """Load fig4.py robustly and expose its estimator helpers."""
    try:
        import base_utils # type: ignore
        return base_utils
    except Exception:
        pass

    path = _THIS_DIR / "base_utils.py"
    if path.exists():
        spec = importlib.util.spec_from_file_location("fig4_loaded", str(path))
        if spec is None or spec.loader is None:
            raise ImportError("Could not load fig4.py.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module

    raise ImportError("Could not locate fig4.py in the script folder.")


fig4_mod = _load_fig4_module()
fig4_prepare_grid_data = fig4_mod.prepare_grid_data
fig4_fit_condfree_from_grid = fig4_mod.fit_condfree_from_grid
fig4_fit_ours_cvec_from_grid = fig4_mod.fit_ours_cvec_from_grid

try:
    from partially_global_frechet import (
        fit_pgfr_baseline_from_samples,
        quantile_grid as pgfr_quantile_grid,
    )
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


def quantile_w1(q_pred: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """Approximate 1D W1 by averaging absolute quantile differences."""
    q_pred = np.asarray(q_pred, dtype=float)
    q_true = np.asarray(q_true, dtype=float)
    if q_pred.shape != q_true.shape:
        raise ValueError(f"q_pred and q_true shapes differ: {q_pred.shape} vs {q_true.shape}")
    return np.mean(np.abs(q_pred - q_true), axis=-1)


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


# ============================================================
# Fixed deterministic alpha for the GOT-style mixed simulation
# ============================================================

def make_fixed_decay_alpha(
    p: int,
    alpha_l2_sq: float = 0.5,
    alpha_decay: float = 0.8,
) -> np.ndarray:
    """Deterministic positive, non-uniform alpha with fixed L2 norm.

    The j-th predictor coefficient is proportional to (j+1)^(-alpha_decay),
    j=0,...,p-1, and is normalized so that sum_j alpha_j^2=alpha_l2_sq.
    This keeps the true signal strength fixed across repeated experiments,
    while giving the predictors distinguishable effects.
    """
    p = int(p)
    if p <= 0:
        raise ValueError("p must be positive.")
    if alpha_l2_sq <= 0:
        raise ValueError("alpha_l2_sq must be positive.")
    idx = np.arange(1, p + 1, dtype=float)
    raw = idx ** (-float(alpha_decay))
    return np.sqrt(float(alpha_l2_sq)) * raw / np.linalg.norm(raw)


def generate_mixed_got_1d_data_fixed_decay_alpha(
    *,
    alpha_decay: float = 0.8,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Call generate_mixed_got_1d_data with deterministic decay alpha.

    The original generator internally calls make_true_alpha(...).  To avoid
    editing got_1d_base.py, we temporarily
    replace that function in generate_mixed_got_1d_data's global namespace.
    Other aspects of the simulation are unchanged.
    """
    old_make_true_alpha = generate_mixed_got_1d_data.__globals__.get("make_true_alpha")

    def _fixed_make_true_alpha(
        p: int,
        alpha_l2_sq: float = 0.5,
        rng: Optional[np.random.Generator] = None,
        kind: str = "deterministic_decay",
        sparsity: Optional[int] = None,
    ) -> np.ndarray:
        return make_fixed_decay_alpha(
            p=p,
            alpha_l2_sq=alpha_l2_sq,
            alpha_decay=alpha_decay,
        )

    generate_mixed_got_1d_data.__globals__["make_true_alpha"] = _fixed_make_true_alpha
    try:
        kwargs = dict(kwargs)
        kwargs["alpha_kind"] = "deterministic_decay"
        kwargs["sparsity"] = None
        data = generate_mixed_got_1d_data(**kwargs)
    finally:
        if old_make_true_alpha is not None:
            generate_mixed_got_1d_data.__globals__["make_true_alpha"] = old_make_true_alpha

    data["alpha_true"] = make_fixed_decay_alpha(
        p=int(kwargs["p"]),
        alpha_l2_sq=float(kwargs["alpha_l2_sq"]),
        alpha_decay=float(alpha_decay),
    )
    if "config" in data:
        data["config"]["alpha_kind"] = "deterministic_decay"
        data["config"]["alpha_decay"] = float(alpha_decay)
        data["config"]["sparsity"] = None
    return data


# ============================================================
# Shared preprocessing for phi_0_1-style OT-map estimators
# ============================================================

def build_ot_training_arrays_from_phi(
    X_dist: np.ndarray,
    Y: np.ndarray,
    m_bins: int = 80,
    x_domain: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bin centers, weights, and empirical OT-map values.

    This deliberately uses `phi_0_1.empirical_ot_map_1d` rather than a local
    reimplementation.
    """
    X_dist = np.asarray(X_dist, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X_dist.ndim != 2 or Y.ndim != 2:
        raise ValueError("X_dist and Y must both have shape (N, n_obs).")
    if X_dist.shape[0] != Y.shape[0]:
        raise ValueError("X_dist and Y must have the same N.")

    N = X_dist.shape[0]
    lo, hi = map(float, x_domain)
    edges = np.linspace(lo, hi, int(m_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    w_ij = np.zeros((N, int(m_bins)), dtype=float)
    y_ij = np.zeros((N, int(m_bins)), dtype=float)

    for i in range(N):
        Q_i, _, _ = phi_empirical_ot_map_1d(X_dist[i], Y[i])
        y_ij[i, :] = Q_i(centers)
        counts, _ = np.histogram(X_dist[i], bins=edges)
        w_ij[i, :] = counts.astype(float) / max(float(len(X_dist[i])), 1.0)

    return centers, w_ij, y_ij


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
        q_pred = self.model_.predict_quantiles(X_dist, C)
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
        }
        if self.model_ is not None:
            out["training_loss"] = float(self.model_.fit_info_.get("training_loss", np.nan))
            out["order_hat"] = _safe_jsonable(self.model_.order_)
            out["alpha_hat"] = _safe_jsonable(self.model_.alpha_)
        return out


# ============================================================
# Method 2: Our structured transport, directly calling fig4.py
# ============================================================

@dataclass
class Fig4StructuredTransportRegressor:
    """Structured transport map using fig4.py's Section 4.2 implementation.

    It calls:
        centers, w_ij, y_ij = fig4.prepare_grid_data(...)
        params, S_hat = fig4.fit_ours_cvec_from_grid(...)

    The recorded fitting time is exactly the time spent by the estimator used
    in fig4.py.
    """
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
# Method 3: GP2022 baseline, ignoring scalars, directly calling fig4.py
# ============================================================

@dataclass
class Fig4GP2022IgnoreScalarRegressor:
    """Strict distribution-on-distribution OT-map baseline.

    This baseline ignores all scalar covariates and directly calls
    `fig4.fit_condfree_from_grid`.  It does NOT call the covariate-dependent structured fitter, so the timing
    is not contaminated by the structured model's extra LS/PAVA iterations.
    """
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
    """Wrapper around partially_global_frechet.fit_pgfr_baseline_from_samples.

    Prediction always calls the external predictor returned by
    fit_pgfr_baseline_from_samples for every test sample. This matches the
    timing convention in fig3.py / fig4.py, where PGFR prediction is performed
    through pgfr_predictor(X_te, c_te, n_pred=...).
    """
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
            fit_order=True,  # standard setting: always use greedy order search
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
    m_true: int = 500,
    m_fit: int = 120,
    m_bins: int = 80,
    alpha_l2_sq: float = 0.5,
    alpha_decay: float = 0.8,
    response_noise_sd: float = 0.02,
    map_amplitude: float = 0.35,
    scalar_noise_sd: float = 0.03,
    scalar_range: Tuple[float, float] = (-0.5, 0.5),
    got_order_maxiter: int = 40,
    got_final_maxiter: int = 100,
    tw_bandwidth: Optional[float] = None,
    tw_bandwidth_scale: float = 1.0,
    tw_pred_sample_size: Optional[int] = None,
    verbose: bool = False,
) -> List[Dict[str, object]]:
    """Generate one dataset and evaluate all methods on exactly the same split."""
    # Make all global-np randomness reproducible as well.  The imported GOT
    # generator mostly uses a local Generator, but setting the global seed makes
    # this wrapper robust to any helper using np.random internally.
    np.random.seed(int(seed))

    data = generate_mixed_got_1d_data_fixed_decay_alpha(
        N_train=N_train,
        N_test=N_test,
        p=p,
        n_obs=n_obs,
        m_true=m_true,
        alpha_l2_sq=alpha_l2_sq,
        alpha_decay=alpha_decay,
        map_amplitude=map_amplitude,
        response_noise_sd=response_noise_sd,
        scalar_noise_sd=scalar_noise_sd,
        scalar_range=scalar_range,
        deterministic_scalar_noise=True,
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
    m_true: int = 500,
    m_fit: int = 120,
    m_bins: int = 80,
    alpha_l2_sq: float = 0.5,
    alpha_decay: float = 0.8,
    response_noise_sd: float = 0.02,
    map_amplitude: float = 0.35,
    scalar_noise_sd: float = 0.03,
    scalar_range: Tuple[float, float] = (-0.5, 0.5),
    base_seed: int = 2026,
    got_order_maxiter: int = 40,
    got_final_maxiter: int = 100,
    tw_bandwidth: Optional[float] = None,
    tw_bandwidth_scale: float = 1.0,
    tw_pred_sample_size: Optional[int] = None,
    output_mat: str = "mixed_methods_results.mat",
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
                m_true=m_true,
                m_fit=m_fit,
                m_bins=m_bins,
                alpha_l2_sq=alpha_l2_sq,
                alpha_decay=alpha_decay,
                response_noise_sd=response_noise_sd,
                map_amplitude=map_amplitude,
                scalar_noise_sd=scalar_noise_sd,
                scalar_range=scalar_range,
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
    torch.save(results, output_path)

    if verbose:
        print(f"\nSaved MATLAB results to: {output_path.resolve()}")

    return results


# ============================================================
# Command-line interface
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Compare mixed-predictor distribution regression methods.")
    parser.add_argument("--p-list", type=int, nargs="+", default=[2, 6, 11],
                        help="Total predictor counts p. There is 1 distribution predictor and p-1 scalars.")
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-obs", type=int, default=100)
    parser.add_argument("--m-true", type=int, default=500)
    parser.add_argument("--m-fit", type=int, default=120)
    parser.add_argument("--m-bins", type=int, default=80)
    parser.add_argument("--alpha-l2-sq", type=float, default=2.0)
    parser.add_argument("--alpha-decay", type=float, default=0.8,
                        help="Deterministic alpha rule: alpha_j is proportional to (j+1)^(-alpha_decay), then normalized.")
    parser.add_argument("--response-noise-sd", type=float, default=0.2)
    parser.add_argument("--map-amplitude", type=float, default=0.35)
    parser.add_argument("--scalar-noise-sd", type=float, default=0.03)
    parser.add_argument("--scalar-range", type=float, nargs=2, default=[0, 1.5])
    parser.add_argument("--base-seed", type=int, default=2026)

    # Standard GOT setting: greedy order search is always used.
    parser.add_argument("--got-order-maxiter", type=int, default=100)
    parser.add_argument("--got-final-maxiter", type=int, default=100)

    # Standard TW2025 setting: use the external PGFR predictor.
    parser.add_argument("--tw-bandwidth", type=float, default=None)
    parser.add_argument("--tw-bandwidth-scale", type=float, default=0.1)
    parser.add_argument("--tw-pred-sample-size", type=int, default=None,
                        help="Pseudo-sample size used by the external PGFR predictor at each test point. "
                             "If omitted, use n_obs. For timing comparable to older fig3.py W2 experiments, try 1000.")

    parser.add_argument("--output", type=str, default="results/mixed_methods_alpha_seed_results.mat")
    return parser.parse_args()


def main():
    args = parse_args()

    return run_p_grid_experiment(
        p_list=args.p_list,
        reps=args.reps,
        N_train=args.n_train,
        N_test=args.n_test,
        n_obs=args.n_obs,
        m_true=args.m_true,
        m_fit=args.m_fit,
        m_bins=args.m_bins,
        alpha_l2_sq=args.alpha_l2_sq,
        alpha_decay=args.alpha_decay,
        response_noise_sd=args.response_noise_sd,
        map_amplitude=args.map_amplitude,
        scalar_noise_sd=args.scalar_noise_sd,
        scalar_range=(float(args.scalar_range[0]), float(args.scalar_range[1])),
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
