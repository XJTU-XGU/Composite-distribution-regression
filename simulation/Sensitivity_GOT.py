#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Fixed-p mixed-predictor simulation with corrected data generation.

This script keeps the estimators, fitting settings, timing convention, and
result structure from

    compare_mixed_methods_p_grid_no_flags_fig4_v2_fixed_alpha_seed_50_parallel.py

but uses the following data-generating mechanism for the single setting
p = 11 and N_train = 100:

1. Distributional predictor
   mu_i is a random mixture of three independent Beta components, exactly as in
   the fig3/phi_0_1 simulation:

       f_{mu_i}(x) = sum_{l=1}^3 pi_l Beta(x; a_{il}, b_{il}),
       a_{il}, b_{il} iid Uniform(1, 10).

2. Scalar predictors
   C_i is sampled uniformly from [0, 1.5]^{10}, with no dimension-dependent
   rescaling.

3. Noiseless response
   The original mixed-GOT ordered composition model is retained.  Predictor 0
   is represented by the optimal map from a fixed Monte Carlo approximation of
   the population Wasserstein barycenter of the random Beta-mixture family to
   mu_i.  Each scalar is represented by the same Gaussian-smoothed translation
   map used in the original mixed-GOT simulation.  The true coefficients are
   deterministic polynomial-decay coefficients with fixed squared L2 norm.

   If G_i denotes the ordered composition of the scaled predictor maps, then

       nu_i^0 = (G_i)_# bar_nu,

   where bar_nu is Uniform(0,1), as in the original mixed-GOT generator.

4. Response noise
   The final response uses the random monotone noise map from Structured_transport_map_core.py:

       nu_i = (S_{eps_i})_# nu_i^0.

The script uses replicate-level process parallelism and saves the same nested
result structure:

    results["p_11"][method] = {
        "w1":  [mean test W1 for each successful replicate],
        "time": total training-plus-testing time,
    }

Data-generation time is excluded from each method's reported time, matching the
original comparison script.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

# Avoid nested thread oversubscription inside process workers.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import torch

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

try:
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
except Exception:
    pass

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

P_FIXED = 11
N_TRAIN_FIXED = 100
RESULT_KEY = "p_11"


# ============================================================
# Robust imports of the original comparison code and Structured_transport_map_core.py
# ============================================================

def _load_module_from_candidates(module_name: str, filenames: Sequence[str]):
    try:
        return __import__(module_name)
    except Exception:
        pass

    for filename in filenames:
        path = _THIS_DIR / filename
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module

    raise ImportError(
        f"Could not load {module_name}. Tried files: {', '.join(filenames)}"
    )


base = _load_module_from_candidates(
    "mixed_methods_fixed_alpha_base",
    ("got_experiment.py",),
)

# The original comparison module already loads Structured_transport_map_core.py.  Reuse that exact
# module when possible so the noise implementation is identical.
phi = getattr(base, "phi_mod", None)
if phi is None:
    phi = getattr(base, "phi", None)
if phi is None:
    phi = _load_module_from_candidates(
        "phi_0_1",
        ("Structured_transport_map_core.py", "phi_0_1(1).py", "phi_0_1(2).py"),
    )


# ============================================================
# Quantile and transport utilities
# ============================================================

_STD_NORMAL = NormalDist()


def _ensure_increasing(q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    q = np.asarray(q, dtype=float).copy()
    q = np.maximum.accumulate(q)
    for k in range(1, len(q)):
        if q[k] <= q[k - 1]:
            q[k] = q[k - 1] + eps
    return q


def _normal_quantiles(u: np.ndarray) -> np.ndarray:
    return np.array([_STD_NORMAL.inv_cdf(float(v)) for v in u], dtype=float)


def _interp_with_linear_extrap(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
) -> np.ndarray:
    """Monotone interpolation with linear extrapolation at both boundaries."""
    x_arr = np.asarray(x, dtype=float)
    xp_arr = _ensure_increasing(np.asarray(xp, dtype=float))
    fp_arr = _ensure_increasing(np.asarray(fp, dtype=float))

    y = np.asarray(np.interp(x_arr, xp_arr, fp_arr), dtype=float)
    if len(xp_arr) < 2:
        return y

    left = x_arr < xp_arr[0]
    right = x_arr > xp_arr[-1]
    slope_left = (fp_arr[1] - fp_arr[0]) / max(xp_arr[1] - xp_arr[0], 1e-12)
    slope_right = (fp_arr[-1] - fp_arr[-2]) / max(xp_arr[-1] - xp_arr[-2], 1e-12)
    y[left] = fp_arr[0] + slope_left * (x_arr[left] - xp_arr[0])
    y[right] = fp_arr[-1] + slope_right * (x_arr[right] - xp_arr[-1])
    return y


def _apply_scaled_map(
    x: np.ndarray,
    q_source: np.ndarray,
    q_target: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Apply id + alpha * (T-id), where T maps q_source to q_target."""
    Tx = _interp_with_linear_extrap(x, q_source, q_target)
    return np.asarray(x, dtype=float) + float(alpha) * (Tx - np.asarray(x, dtype=float))


def _sample_from_quantile(
    q_grid: np.ndarray,
    u_grid: np.ndarray,
    n_obs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    uu = rng.uniform(float(u_grid[0]), float(u_grid[-1]), size=int(n_obs))
    return np.interp(uu, u_grid, q_grid)


# ============================================================
# Beta-mixture predictor generation and fixed reference barycenter
# ============================================================

def _draw_beta_mixture_parameters(
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    alphas = rng.uniform(1.0, 10.0, size=3)
    betas = rng.uniform(1.0, 10.0, size=3)
    return alphas, betas


def _sample_beta_mixture_local(
    rng: np.random.Generator,
    alphas: np.ndarray,
    betas: np.ndarray,
    pis: Tuple[float, float, float],
    size: int,
) -> np.ndarray:
    components = rng.choice(3, size=int(size), p=np.asarray(pis, dtype=float))
    return rng.beta(alphas[components], betas[components])


@functools.lru_cache(maxsize=8)
def _population_beta_mixture_barycenter_quantile(
    m_true: int,
    pis: Tuple[float, float, float],
    reference_distributions: int,
    reference_n_obs: int,
    reference_seed: int,
) -> np.ndarray:
    """Deterministic Monte Carlo approximation of E[Q_mu(u)].

    This is computed from an auxiliary sample independent of every train/test
    replicate, so the true reference distribution is fixed across replications
    and does not use test information.
    """
    rng = np.random.default_rng(int(reference_seed))
    u_grid = base.make_quantile_grid(int(m_true))
    q_sum = np.zeros(int(m_true), dtype=float)

    for _ in range(int(reference_distributions)):
        alphas, betas = _draw_beta_mixture_parameters(rng)
        sample = _sample_beta_mixture_local(
            rng=rng,
            alphas=alphas,
            betas=betas,
            pis=pis,
            size=int(reference_n_obs),
        )
        q_sum += np.quantile(sample, u_grid)

    q_bar = q_sum / float(reference_distributions)
    return _ensure_increasing(q_bar)


def _fixed_decay_alpha(
    p: int,
    alpha_l2_sq: float,
    alpha_decay: float,
) -> np.ndarray:
    if hasattr(base, "make_fixed_decay_alpha"):
        return np.asarray(
            base.make_fixed_decay_alpha(
                p=int(p),
                alpha_l2_sq=float(alpha_l2_sq),
                alpha_decay=float(alpha_decay),
            ),
            dtype=float,
        )

    idx = np.arange(1, int(p) + 1, dtype=float)
    raw = idx ** (-float(alpha_decay))
    return np.sqrt(float(alpha_l2_sq)) * raw / np.linalg.norm(raw)


# ============================================================
# Corrected hybrid data generator
# ============================================================

def generate_corrected_p11_data(
    N_train: int = N_TRAIN_FIXED,
    N_test: int = 1000,
    p: int = P_FIXED,
    n_obs: int = 100,
    m_true: int = 500,
    pis: Tuple[float, float, float] = (0.3, 0.4, 0.3),
    scalar_range: Tuple[float, float] = (0.0, 1.5),
    scalar_noise_sd: float = 0.03,
    alpha_l2_sq: float = 2.0,
    alpha_decay: float = 0.8,
    J_segments: int = 8,
    noise_domain_min: float = 0.0,
    noise_domain_max: float = 4.0,
    noise_domain_buffer: float = 0.25,
    reference_distributions: int = 1000,
    reference_n_obs: int = 1000,
    reference_seed: int = 314159,
    seed: int = 2026,
) -> Dict[str, np.ndarray]:
    """Generate the corrected p=11, N_train=100 simulation dataset."""
    if int(p) != P_FIXED:
        raise ValueError(f"This script is fixed at p={P_FIXED}.")
    if int(N_train) != N_TRAIN_FIXED:
        raise ValueError(f"This script is fixed at N_train={N_TRAIN_FIXED}.")
    if len(pis) != 3 or not np.isclose(sum(pis), 1.0):
        raise ValueError("pis must contain three nonnegative weights summing to one.")
    if scalar_range[0] >= scalar_range[1]:
        raise ValueError("scalar_range must satisfy lower < upper.")

    np.random.seed(int(seed))
    rng = np.random.default_rng(int(seed))

    p = int(p)
    d = p - 1
    u_grid = base.make_quantile_grid(int(m_true))
    z_grid = _normal_quantiles(u_grid)

    alpha_true = _fixed_decay_alpha(
        p=p,
        alpha_l2_sq=float(alpha_l2_sq),
        alpha_decay=float(alpha_decay),
    )
    true_order = np.arange(p, dtype=int)

    # Predictor-0 reference: fixed population Wasserstein barycenter of the
    # random Beta-mixture family, approximated independently of the experiment.
    q_mu0_ref = _population_beta_mixture_barycenter_quantile(
        int(m_true),
        tuple(float(v) for v in pis),
        int(reference_distributions),
        int(reference_n_obs),
        int(reference_seed),
    ).copy()

    # Scalar references are the Gaussian-smoothed distributions centered at the
    # midpoint of the scalar range, exactly as in the original mixed generator.
    scalar_center = 0.5 * (float(scalar_range[0]) + float(scalar_range[1]))
    q_scalar_ref = _ensure_increasing(
        scalar_center + float(scalar_noise_sd) * z_grid
    )

    # The original response reference distribution is Uniform(0,1).
    q_nu_ref = u_grid.copy()

    def _generate_predictors(N: int):
        X_dist = np.empty((int(N), int(n_obs)), dtype=float)
        C = rng.uniform(
            float(scalar_range[0]),
            float(scalar_range[1]),
            size=(int(N), d),
        )
        qX0 = np.empty((int(N), int(m_true)), dtype=float)

        for i in range(int(N)):
            # Use the exact fig3/phi_0_1 primitives for the observed predictor.
            alphas, betas = phi.sample_mu_params()
            Xi = phi.sample_from_mixture_beta(
                alphas,
                betas,
                pis,
                size=int(n_obs),
            )
            Xi = np.asarray(Xi, dtype=float)
            X_dist[i] = Xi
            qX0[i] = _ensure_increasing(np.quantile(Xi, u_grid))

        return X_dist, C, qX0

    X_train, C_train, qX0_train = _generate_predictors(int(N_train))
    X_test, C_test, qX0_test = _generate_predictors(int(N_test))

    def _make_noiseless_quantiles(C: np.ndarray, qX0: np.ndarray) -> np.ndarray:
        qY0 = np.empty((C.shape[0], int(m_true)), dtype=float)

        for i in range(C.shape[0]):
            q_targets: List[np.ndarray] = [qX0[i]]
            q_sources: List[np.ndarray] = [q_mu0_ref]

            for k in range(d):
                q_target = _ensure_increasing(
                    float(C[i, k]) + float(scalar_noise_sd) * z_grid
                )
                q_targets.append(q_target)
                q_sources.append(q_scalar_ref)

            q = q_nu_ref.copy()
            # Preserve the original ordered-composition convention: the
            # rightmost map acts first, hence reversed(true_order).
            for j in reversed(true_order):
                q = _apply_scaled_map(
                    q,
                    q_source=q_sources[int(j)],
                    q_target=q_targets[int(j)],
                    alpha=float(alpha_true[int(j)]),
                )

            qY0[i] = _ensure_increasing(q)

        return qY0

    qY0_train = _make_noiseless_quantiles(C_train, qX0_train)
    qY0_test = _make_noiseless_quantiles(C_test, qX0_test)

    all_min = min(float(np.min(qY0_train)), float(np.min(qY0_test)))
    all_max = max(float(np.max(qY0_train)), float(np.max(qY0_test)))
    domain_min_used = min(
        float(noise_domain_min),
        all_min - float(noise_domain_buffer),
    )
    domain_max_used = max(
        float(noise_domain_max),
        all_max + float(noise_domain_buffer),
    )

    def _add_fig3_noise_and_sample(qY0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Y = np.empty((qY0.shape[0], int(n_obs)), dtype=float)
        qY = np.empty_like(qY0)

        for i in range(qY0.shape[0]):
            S_eps = phi.sample_random_S_eps(
                J=int(J_segments),
                domain_min=float(domain_min_used),
                domain_max=float(domain_max_used),
            )
            qi = _ensure_increasing(np.asarray(S_eps(qY0[i]), dtype=float))
            qY[i] = qi
            Y[i] = _sample_from_quantile(qi, u_grid, int(n_obs), rng)

        return Y, qY

    Y_train, qY_train = _add_fig3_noise_and_sample(qY0_train)
    Y_test, qY_test = _add_fig3_noise_and_sample(qY0_test)

    return {
        "X_train_dist": X_train,
        "Y_train": Y_train,
        "C_train_scalar": C_train,
        "X_test_dist": X_test,
        "Y_test": Y_test,
        "C_test_scalar": C_test,
        "alpha_true": alpha_true,
        "true_order": true_order,
        "u_grid_true": u_grid,
        "q_mu0_reference": q_mu0_ref,
        "q_nu_reference": q_nu_ref,
        "qX0_train_true": qX0_train,
        "qX0_test_true": qX0_test,
        "qY0_train_true": qY0_train,
        "qY0_test_true": qY0_test,
        "qY_train_true": qY_train,
        "qY_test_true": qY_test,
        "config": {
            "data_source": "beta_mixture_mu_uniform_scalar_got_composition_fig3_noise",
            "N_train": int(N_train),
            "N_test": int(N_test),
            "p": int(p),
            "n_scalar": int(d),
            "n_obs": int(n_obs),
            "m_true": int(m_true),
            "pis": list(pis),
            "scalar_range": list(scalar_range),
            "scalar_noise_sd": float(scalar_noise_sd),
            "alpha_l2_sq": float(alpha_l2_sq),
            "alpha_decay": float(alpha_decay),
            "J_segments": int(J_segments),
            "noise_domain_min_used": float(domain_min_used),
            "noise_domain_max_used": float(domain_max_used),
            "reference_distributions": int(reference_distributions),
            "reference_n_obs": int(reference_n_obs),
            "reference_seed": int(reference_seed),
            "seed": int(seed),
        },
    }


# ============================================================
# Method fitting and replicate-level parallelism
# ============================================================

def run_one_replicate(
    rep: int,
    seed: int,
    N_test: int = 1000,
    n_obs: int = 100,
    m_true: int = 500,
    m_fit: int = 120,
    m_bins: int = 80,
    pis: Tuple[float, float, float] = (0.3, 0.4, 0.3),
    scalar_range: Tuple[float, float] = (0.0, 1.5),
    scalar_noise_sd: float = 0.03,
    alpha_l2_sq: float = 2.0,
    alpha_decay: float = 0.8,
    J_segments: int = 8,
    noise_domain_min: float = 0.0,
    noise_domain_max: float = 4.0,
    noise_domain_buffer: float = 0.25,
    reference_distributions: int = 1000,
    reference_n_obs: int = 1000,
    reference_seed: int = 314159,
    got_order_maxiter: int = 100,
    got_final_maxiter: int = 100,
    tw_bandwidth: Optional[float] = None,
    tw_bandwidth_scale: float = 0.1,
    tw_pred_sample_size: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if verbose:
        print(
            f"[rep={rep + 1}, seed={seed}] generating data "
            f"(N_train={N_TRAIN_FIXED}, N_test={int(N_test)}) ...",
            flush=True,
        )
    data = generate_corrected_p11_data(
        N_train=N_TRAIN_FIXED,
        N_test=int(N_test),
        p=P_FIXED,
        n_obs=int(n_obs),
        m_true=int(m_true),
        pis=pis,
        scalar_range=scalar_range,
        scalar_noise_sd=float(scalar_noise_sd),
        alpha_l2_sq=float(alpha_l2_sq),
        alpha_decay=float(alpha_decay),
        J_segments=int(J_segments),
        noise_domain_min=float(noise_domain_min),
        noise_domain_max=float(noise_domain_max),
        noise_domain_buffer=float(noise_domain_buffer),
        reference_distributions=int(reference_distributions),
        reference_n_obs=int(reference_n_obs),
        reference_seed=int(reference_seed),
        seed=int(seed),
    )
    if verbose:
        print(f"[rep={rep + 1}, seed={seed}] data generated.", flush=True)

    methods = base.make_methods(
        p=P_FIXED,
        seed=int(seed),
        m_fit=int(m_fit),
        m_bins=int(m_bins),
        scalar_noise_sd=float(scalar_noise_sd),
        got_order_maxiter=int(got_order_maxiter),
        got_final_maxiter=int(got_final_maxiter),
        tw_bandwidth=tw_bandwidth,
        tw_bandwidth_scale=float(tw_bandwidth_scale),
        tw_pred_sample_size=tw_pred_sample_size,
        verbose=verbose,
    )
    u_grid_eval = base.make_quantile_grid(int(m_fit))

    rows: List[Dict[str, object]] = []
    for method_index, (method_name, model) in enumerate(methods.items()):
        # Keep method-side randomness reproducible and independent of method order.
        method_seed = int(seed + 10000 + 1000 * method_index)
        np.random.seed(method_seed)
        torch.manual_seed(method_seed)

        try:
            if verbose:
                print(
                    f"[rep={rep + 1}, seed={seed}] fitting {method_name} ...",
                    flush=True,
                )
            row = base.fit_predict_score_method(
                method_name=method_name,
                model=model,
                data=data,
                u_grid_eval=u_grid_eval,
            )
            row["status"] = "ok"
            rows.append(row)
            if verbose:
                print(
                    f"[rep={rep + 1}, seed={seed}] finished {method_name}: "
                    f"mean W1={float(row['w1']):.4e}, "
                    f"time={float(row['total_train_test_time']):.2f}s",
                    flush=True,
                )
        except Exception as exc:
            rows.append(
                {
                    "method": method_name,
                    "w1": np.nan,
                    "total_train_test_time": np.nan,
                    "status": "failed",
                    "error": repr(exc),
                }
            )
            if verbose:
                print(
                    f"[rep={rep + 1}, seed={seed}] {method_name} failed: {exc!r}",
                    flush=True,
                )

    return rows, dict(data["config"])


def _worker(task: Dict[str, object]) -> Dict[str, object]:
    task = dict(task)
    rep = int(task.pop("rep"))
    seed = int(task.pop("seed"))
    task["verbose"] = bool(task.get("verbose", True))
    rows, data_config = run_one_replicate(rep=rep, seed=seed, **task)
    return {
        "rep": rep,
        "seed": seed,
        "rows": rows,
        "data_config": data_config,
    }


def _accumulate(
    setting_results: Dict[str, Dict[str, object]],
    rows: List[Dict[str, object]],
) -> None:
    for row in rows:
        method_name = str(row["method"])
        result = setting_results.setdefault(
            method_name,
            {"w1": [], "time": 0.0, "n_failed": 0, "errors": []},
        )
        if row["status"] == "ok":
            result["w1"].append(float(row["w1"]))
            result["time"] += float(row["total_train_test_time"])
        else:
            result["n_failed"] += 1
            result["errors"].append(str(row.get("error", "unknown error")))


def _print_rows(rep: int, reps: int, seed: int, rows: List[Dict[str, object]]) -> None:
    print(f"[rep={rep + 1}/{reps}, p=11, N=100, seed={seed}]", flush=True)
    for row in rows:
        if row["status"] == "ok":
            print(
                f"  {row['method']}: mean W1={float(row['w1']):.4e}, "
                f"time={float(row['total_train_test_time']):.2f}s",
                flush=True,
            )
        else:
            print(f"  {row['method']}: failed: {row.get('error', '')}", flush=True)


def _save_results(results: Dict[str, object], output_file: str) -> Path:
    path = Path(output_file)
    if path.suffix.lower() not in {".mat", ".pt", ".pth"}:
        path = path.with_suffix(".mat")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, path)
    return path


def run_experiment(
    reps: int = 100,
    N_test: int = 1000,
    n_obs: int = 100,
    m_true: int = 500,
    m_fit: int = 120,
    m_bins: int = 80,
    pis: Tuple[float, float, float] = (0.3, 0.4, 0.3),
    scalar_range: Tuple[float, float] = (0.0, 1.5),
    scalar_noise_sd: float = 0.03,
    alpha_l2_sq: float = 2.0,
    alpha_decay: float = 0.8,
    J_segments: int = 8,
    noise_domain_min: float = 0.0,
    noise_domain_max: float = 4.0,
    noise_domain_buffer: float = 0.25,
    reference_distributions: int = 1000,
    reference_n_obs: int = 1000,
    reference_seed: int = 314159,
    base_seed: int = 2026,
    got_order_maxiter: int = 100,
    got_final_maxiter: int = 100,
    tw_bandwidth: Optional[float] = None,
    tw_bandwidth_scale: float = 0.1,
    tw_pred_sample_size: Optional[int] = None,
    output_file: str = "results_got/mixed_methods_p11_N100_beta_mu_fig3_noise_fixed_alpha.mat",
    workers: int = 4,
    print_results: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    results: Dict[str, object] = {
        RESULT_KEY: {},
        "config": {
            "experiment": "p11_N100_beta_mu_got_composition_fig3_noise",
            "p": P_FIXED,
            "N_train": N_TRAIN_FIXED,
            "N_test": int(N_test),
            "reps": int(reps),
            "n_obs": int(n_obs),
            "m_true": int(m_true),
            "m_fit": int(m_fit),
            "m_bins": int(m_bins),
            "pis": list(pis),
            "scalar_range": list(scalar_range),
            "scalar_noise_sd": float(scalar_noise_sd),
            "alpha_l2_sq": float(alpha_l2_sq),
            "alpha_decay": float(alpha_decay),
            "J_segments": int(J_segments),
            "noise_domain_min": float(noise_domain_min),
            "noise_domain_max": float(noise_domain_max),
            "noise_domain_buffer": float(noise_domain_buffer),
            "reference_distributions": int(reference_distributions),
            "reference_n_obs": int(reference_n_obs),
            "reference_seed": int(reference_seed),
            "base_seed": int(base_seed),
            "workers": int(workers),
        },
        "replicate_data_diagnostics": [],
    }
    setting_results = results[RESULT_KEY]
    assert isinstance(setting_results, dict)

    common: Dict[str, object] = {
        "N_test": int(N_test),
        "n_obs": int(n_obs),
        "m_true": int(m_true),
        "m_fit": int(m_fit),
        "m_bins": int(m_bins),
        "pis": pis,
        "scalar_range": scalar_range,
        "scalar_noise_sd": float(scalar_noise_sd),
        "alpha_l2_sq": float(alpha_l2_sq),
        "alpha_decay": float(alpha_decay),
        "J_segments": int(J_segments),
        "noise_domain_min": float(noise_domain_min),
        "noise_domain_max": float(noise_domain_max),
        "noise_domain_buffer": float(noise_domain_buffer),
        "reference_distributions": int(reference_distributions),
        "reference_n_obs": int(reference_n_obs),
        "reference_seed": int(reference_seed),
        "got_order_maxiter": int(got_order_maxiter),
        "got_final_maxiter": int(got_final_maxiter),
        "tw_bandwidth": tw_bandwidth,
        "tw_bandwidth_scale": float(tw_bandwidth_scale),
        "tw_pred_sample_size": tw_pred_sample_size,
    }

    worker_count = max(1, min(int(workers), int(reps)))

    if worker_count == 1:
        for rep in range(int(reps)):
            seed = int(base_seed + 100000 * P_FIXED + rep)
            rows, data_config = run_one_replicate(
                rep=rep,
                seed=seed,
                verbose=verbose,
                **common,
            )
            _accumulate(setting_results, rows)
            diagnostics = results["replicate_data_diagnostics"]
            assert isinstance(diagnostics, list)
            diagnostics.append(
                {
                    "rep": rep,
                    "seed": seed,
                    "noise_domain_min_used": data_config["noise_domain_min_used"],
                    "noise_domain_max_used": data_config["noise_domain_max_used"],
                }
            )
            if print_results:
                _print_rows(rep, int(reps), seed, rows)
            _save_results(results, output_file)
    else:
        if verbose:
            print(
                f"Running {int(reps)} replicates for p=11, N=100 with "
                f"{worker_count} worker processes.",
                flush=True,
            )

        futures = {}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for rep in range(int(reps)):
                seed = int(base_seed + 100000 * P_FIXED + rep)
                task = dict(common)
                task.update({"rep": rep, "seed": seed})
                futures[executor.submit(_worker, task)] = (rep, seed)

            completed: Dict[int, Dict[str, object]] = {}
            for future in as_completed(futures):
                rep, seed = futures[future]
                try:
                    payload = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Parallel replicate failed for rep={rep}, seed={seed}."
                    ) from exc

                completed[int(payload["rep"])] = payload
                if print_results:
                    _print_rows(
                        int(payload["rep"]),
                        int(reps),
                        int(payload["seed"]),
                        payload["rows"],  # type: ignore[arg-type]
                    )

        # Accumulate in replicate order so the stored W1 vectors are deterministic.
        for rep in sorted(completed):
            payload = completed[rep]
            rows = payload["rows"]
            _accumulate(setting_results, rows)  # type: ignore[arg-type]
            data_config = payload["data_config"]
            diagnostics = results["replicate_data_diagnostics"]
            assert isinstance(diagnostics, list)
            diagnostics.append(
                {
                    "rep": int(payload["rep"]),
                    "seed": int(payload["seed"]),
                    "noise_domain_min_used": data_config["noise_domain_min_used"],
                    "noise_domain_max_used": data_config["noise_domain_max_used"],
                }
            )

        _save_results(results, output_file)

    final_path = _save_results(results, output_file)
    if verbose:
        print(f"\nSaved results to: {final_path.resolve()}", flush=True)
        for method_name, method_result in setting_results.items():
            values = np.asarray(method_result.get("w1", []), dtype=float)
            total_time = float(method_result.get("time", 0.0))
            failed = int(method_result.get("n_failed", 0))
            if len(values):
                print(
                    f"  {method_name}: W1={np.mean(values):.4e} +/- "
                    f"{np.std(values):.4e}, total time={total_time:.2f}s, "
                    f"failed={failed}",
                    flush=True,
                )
            else:
                print(f"  {method_name}: no successful runs, failed={failed}", flush=True)

    return results


# ============================================================
# Command-line interface
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Corrected p=11, N_train=100 mixed-predictor experiment with "
            "Beta-mixture mu, uniform scalars, fixed-decay GOT composition, "
            "and fig3 random-map response noise."
        )
    )
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--N-test", "--n-test", type=int, default=1000)
    parser.add_argument("--n-obs", type=int, default=100)
    parser.add_argument("--m-true", type=int, default=500)
    parser.add_argument("--m-fit", type=int, default=120)
    parser.add_argument("--m-bins", type=int, default=80)
    parser.add_argument("--pis", type=float, nargs=3, default=[0.3, 0.4, 0.3])
    parser.add_argument("--scalar-range", type=float, nargs=2, default=[0.0, 1.5])
    parser.add_argument("--scalar-noise-sd", type=float, default=0.03)
    parser.add_argument("--alpha-l2-sq", type=float, default=2.0)
    parser.add_argument("--alpha-decay", type=float, default=0.8)
    parser.add_argument("--J-segments", type=int, default=8)
    parser.add_argument("--noise-domain-min", type=float, default=0.0)
    parser.add_argument("--noise-domain-max", type=float, default=4.0)
    parser.add_argument("--noise-domain-buffer", type=float, default=0.25)
    parser.add_argument("--reference-distributions", type=int, default=1000)
    parser.add_argument("--reference-n-obs", type=int, default=1000)
    parser.add_argument("--reference-seed", type=int, default=314159)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--got-order-maxiter", type=int, default=100)
    parser.add_argument("--got-final-maxiter", type=int, default=100)
    parser.add_argument("--tw-bandwidth", type=float, default=None)
    parser.add_argument("--tw-bandwidth-scale", type=float, default=0.1)
    parser.add_argument("--tw-pred-sample-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--print-results", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--output",
        type=str,
        default="results_got/mixed_methods_p11_N100_beta_mu_fig3_noise_fixed_alpha.mat",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    return run_experiment(
        reps=args.reps,
        N_test=args.N_test,
        n_obs=args.n_obs,
        m_true=args.m_true,
        m_fit=args.m_fit,
        m_bins=args.m_bins,
        pis=(float(args.pis[0]), float(args.pis[1]), float(args.pis[2])),
        scalar_range=(float(args.scalar_range[0]), float(args.scalar_range[1])),
        scalar_noise_sd=args.scalar_noise_sd,
        alpha_l2_sq=args.alpha_l2_sq,
        alpha_decay=args.alpha_decay,
        J_segments=args.J_segments,
        noise_domain_min=args.noise_domain_min,
        noise_domain_max=args.noise_domain_max,
        noise_domain_buffer=args.noise_domain_buffer,
        reference_distributions=args.reference_distributions,
        reference_n_obs=args.reference_n_obs,
        reference_seed=args.reference_seed,
        base_seed=args.base_seed,
        got_order_maxiter=args.got_order_maxiter,
        got_final_maxiter=args.got_final_maxiter,
        tw_bandwidth=args.tw_bandwidth,
        tw_bandwidth_scale=args.tw_bandwidth_scale,
        tw_pred_sample_size=args.tw_pred_sample_size,
        output_file=args.output,
        workers=args.workers,
        print_results=bool(args.print_results),
        verbose=True,
    )


if __name__ == "__main__":
    import multiprocessing as _multiprocessing

    _multiprocessing.freeze_support()
    main()
