#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Misspecified scalar-covariate experiment.

The true regression map is independent of scalar predictors, corresponding to
the p=1, d=0 scientific setting.  During fitting, however, we still pass 10
irrelevant scalar covariates to the structured transport estimator, as if
p=11 and d=10 were needed.

Only two methods are compared:

1. structured_transport
   Our method, fit with the irrelevant d=10 scalar predictors.

2. GP2022_condfree_OT
   Ghodrati & Panaretos (2022), a distribution-only baseline that ignores
   scalar predictors.

Saved result format matches the existing torch-saved .mat files:

    results["p_11_N_100"][method] = {
        "w1": [mean W1 from each successful replicate],
        "time": total training-and-testing time across successful replicates,
    }
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

import Grid_d_N100 as base


METHOD_NAMES = (
    "structured_transport",
    "GP2022_condfree_OT",
)


def S0_true_scalar_irrelevant(
    x: np.ndarray,
    truth_phi0: float = 1.0,
    truth_phi1: float = 0.5,
) -> np.ndarray:
    """True map with no scalar-covariate dependence."""
    x = np.asarray(x, dtype=float)
    return float(truth_phi0) * base.phi.zeta_k(x, k=4) + float(truth_phi1)


def generate_scalar_irrelevant_data(
    N_train: int = 100,
    N_test: int = 1000,
    p_fit: int = 11,
    n_obs: int = 100,
    pis: Tuple[float, float, float] = (0.3, 0.4, 0.3),
    J_segments: int = 8,
    scalar_range: Tuple[float, float] = (0.0, 1.5),
    scalar_scaling: str = "none",
    truth_phi0: float = 1.0,
    truth_phi1: float = 0.5,
    domain_max: float = 4.0,
    domain_buffer: float = 0.25,
    seed: int = 2026,
) -> Dict[str, np.ndarray]:
    """
    Generate data with d_true=0 but d_fit=p_fit-1 scalar covariates recorded.

    C_train_scalar and C_test_scalar are intentionally irrelevant noise
    covariates.  They are included so our method is fit under the misspecified
    d_fit=10 model, while the true response-generating map ignores them.
    """
    np.random.seed(int(seed))
    rng = np.random.default_rng(int(seed))

    p_fit = int(p_fit)
    if p_fit < 1:
        raise ValueError("p_fit must be at least 1.")
    d_fit = p_fit - 1

    s0_upper = float(truth_phi0) + float(truth_phi1)
    domain_max_used = max(float(domain_max), float(s0_upper + domain_buffer))

    def _make_split(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.empty((int(N), int(n_obs)), dtype=float)
        Y = np.empty((int(N), int(n_obs)), dtype=float)
        C = base._scaled_scalar_matrix(
            rng=rng,
            N=int(N),
            d=int(d_fit),
            scalar_range=scalar_range,
            scalar_scaling=scalar_scaling,
        )

        for i in range(int(N)):
            alphas, betas = base.phi.sample_mu_params()
            Xi = base.phi.sample_from_mixture_beta(alphas, betas, pis, size=int(n_obs))
            S_eps = base.phi.sample_random_S_eps(
                J=int(J_segments),
                domain_min=0.0,
                domain_max=float(domain_max_used),
            )
            S0_vals = S0_true_scalar_irrelevant(
                Xi,
                truth_phi0=truth_phi0,
                truth_phi1=truth_phi1,
            )
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
        "truth_phi0": float(truth_phi0),
        "truth_phi1": float(truth_phi1),
        "config": {
            "data_source": "fig3_scalar_irrelevant_truth",
            "truth_setting": "p_true=1_d_true=0",
            "fit_setting": f"p_fit={int(p_fit)}_d_fit={int(d_fit)}",
            "N_train": int(N_train),
            "N_test": int(N_test),
            "p_fit": int(p_fit),
            "d_fit": int(d_fit),
            "n_obs": int(n_obs),
            "pis": list(pis),
            "J_segments": int(J_segments),
            "scalar_range_raw": list(scalar_range),
            "scalar_scaling": scalar_scaling,
            "truth_phi0": float(truth_phi0),
            "truth_phi1": float(truth_phi1),
            "domain_max_requested": float(domain_max),
            "domain_max_used": float(domain_max_used),
            "seed": int(seed),
        },
    }


def make_methods(m_bins: int = 80):
    """Construct fresh method objects for one replicate."""
    return {
        "structured_transport": base.Fig4StructuredTransportRegressor(
            m_bins=m_bins,
            max_iter=80,
            tol=1e-6,
        ),
        "GP2022_condfree_OT": base.Fig4GP2022IgnoreScalarRegressor(
            m_bins=m_bins,
        ),
    }


def run_one_replicate(
    rep: int,
    seed: int,
    p_fit: int = 11,
    N_train: int = 100,
    N_test: int = 1000,
    n_obs: int = 100,
    m_fit: int = 120,
    m_bins: int = 80,
    pis: Tuple[float, float, float] = (0.3, 0.4, 0.3),
    J_segments: int = 8,
    scalar_range: Tuple[float, float] = (0.0, 1.5),
    scalar_scaling: str = "none",
    truth_phi0: float = 1.0,
    truth_phi1: float = 0.5,
    domain_max: float = 4.0,
    domain_buffer: float = 0.25,
    verbose: bool = False,
) -> List[Dict[str, object]]:
    data = generate_scalar_irrelevant_data(
        N_train=N_train,
        N_test=N_test,
        p_fit=p_fit,
        n_obs=n_obs,
        pis=pis,
        J_segments=J_segments,
        scalar_range=scalar_range,
        scalar_scaling=scalar_scaling,
        truth_phi0=truth_phi0,
        truth_phi1=truth_phi1,
        domain_max=domain_max,
        domain_buffer=domain_buffer,
        seed=seed,
    )

    methods = make_methods(m_bins=m_bins)
    u_grid_eval = base.make_quantile_grid(m_fit)

    run_rows: List[Dict[str, object]] = []
    for method_name, model in methods.items():
        try:
            if verbose:
                print(f"    fitting {method_name} ...", flush=True)
            row = base.fit_predict_score_method(
                method_name=method_name,
                model=model,
                data=data,
                u_grid_eval=u_grid_eval,
            )
            row["status"] = "ok"
            run_rows.append(row)
            if verbose:
                print(
                    f"      {method_name}: mean W1={row['w1']:.4e}, "
                    f"time={row['total_train_test_time']:.2f}s",
                    flush=True,
                )
        except Exception as exc:
            run_rows.append({
                "method": method_name,
                "w1": np.nan,
                "total_train_test_time": np.nan,
                "status": "failed",
            })
            print(
                f"[WARNING] {method_name} failed for rep={rep}, "
                f"p_fit={p_fit}, N_train={N_train}: {exc}",
                flush=True,
            )

    return run_rows


def save_results(results: Dict[str, Dict[str, Dict[str, object]]], output_mat: str) -> Path:
    output_path = Path(output_mat)
    if output_path.suffix.lower() != ".mat":
        output_path = output_path.with_suffix(".mat")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, output_path)
    return output_path


def run_experiment(
    reps: int = 100,
    p_fit: int = 11,
    N_train: int = 100,
    N_test: int = 1000,
    n_obs: int = 100,
    m_fit: int = 120,
    m_bins: int = 80,
    pis: Tuple[float, float, float] = (0.3, 0.4, 0.3),
    J_segments: int = 8,
    scalar_range: Tuple[float, float] = (0.0, 1.5),
    scalar_scaling: str = "none",
    truth_phi0: float = 1.0,
    truth_phi1: float = 0.5,
    domain_max: float = 4.0,
    domain_buffer: float = 0.25,
    base_seed: int = 2026,
    output_mat: str = "results_null_scalar/fig3_scalar_irrelevant_assume_d10_results.mat",
    verbose: bool = True,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    setting_key = f"p_{int(p_fit)}_N_{int(N_train)}"
    results: Dict[str, Dict[str, Dict[str, object]]] = {
        setting_key: {
            method_name: {"w1": [], "time": 0.0}
            for method_name in METHOD_NAMES
        }
    }

    for rep in range(int(reps)):
        seed = int(base_seed + 100000 * int(p_fit) + rep)
        if verbose:
            print(
                f"[{setting_key}, d_fit={int(p_fit) - 1}, "
                f"rep={rep + 1}/{reps}, seed={seed}]",
                flush=True,
            )

        run_rows = run_one_replicate(
            rep=int(rep),
            seed=seed,
            p_fit=int(p_fit),
            N_train=int(N_train),
            N_test=int(N_test),
            n_obs=int(n_obs),
            m_fit=int(m_fit),
            m_bins=int(m_bins),
            pis=pis,
            J_segments=int(J_segments),
            scalar_range=scalar_range,
            scalar_scaling=scalar_scaling,
            truth_phi0=float(truth_phi0),
            truth_phi1=float(truth_phi1),
            domain_max=float(domain_max),
            domain_buffer=float(domain_buffer),
            verbose=verbose,
        )

        for row in run_rows:
            if row["status"] != "ok":
                continue
            method_name = str(row["method"])
            method_result = results[setting_key].setdefault(method_name, {"w1": [], "time": 0.0})
            method_result["w1"].append(float(row["w1"]))
            method_result["time"] += float(row["total_train_test_time"])

        save_results(results, output_mat)

    output_path = save_results(results, output_mat)
    if verbose:
        print(f"\nSaved results to: {output_path.resolve()}")
        print_summary(results)
    return results


def print_summary(results: Dict[str, Dict[str, Dict[str, object]]]) -> None:
    for setting_key, method_results in results.items():
        print(f"\nSummary for {setting_key}:")
        for method_name in METHOD_NAMES:
            record = method_results.get(method_name, {})
            w1 = np.asarray(record.get("w1", []), dtype=float)
            if w1.size == 0:
                print(f"  {method_name}: no successful runs")
                continue
            sd = float(np.std(w1, ddof=1)) if w1.size > 1 else 0.0
            print(
                f"  {method_name}: "
                f"w1_len={w1.size}, mean={float(np.mean(w1)):.6g}, "
                f"sd={sd:.6g}, time={float(record.get('time', 0.0)):.6g}s"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "True scalar-irrelevant fig3 experiment: fit as p=11/d=10 and "
            "compare our method against Ghodrati & Panaretos (2022)."
        )
    )
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--p-fit", type=int, default=11)
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-obs", type=int, default=100)
    parser.add_argument("--m-fit", type=int, default=120)
    parser.add_argument("--m-bins", type=int, default=80)
    parser.add_argument("--pis", type=float, nargs=3, default=[0.3, 0.4, 0.3])
    parser.add_argument("--J-segments", type=int, default=8)
    parser.add_argument("--scalar-range", type=float, nargs=2, default=[0.0, 1.5])
    parser.add_argument("--scalar-scaling", type=str, default="none", choices=["none", "sqrt_d"])
    parser.add_argument("--truth-phi0", type=float, default=1.0)
    parser.add_argument("--truth-phi1", type=float, default=0.5)
    parser.add_argument("--domain-max", type=float, default=4.0)
    parser.add_argument("--domain-buffer", type=float, default=0.25)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=str,
        default="results_null_scalar/fig3_scalar_irrelevant_assume_d10_results.mat",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    return run_experiment(
        reps=args.reps,
        p_fit=args.p_fit,
        N_train=args.n_train,
        N_test=args.n_test,
        n_obs=args.n_obs,
        m_fit=args.m_fit,
        m_bins=args.m_bins,
        pis=(float(args.pis[0]), float(args.pis[1]), float(args.pis[2])),
        J_segments=args.J_segments,
        scalar_range=(float(args.scalar_range[0]), float(args.scalar_range[1])),
        scalar_scaling=args.scalar_scaling,
        truth_phi0=args.truth_phi0,
        truth_phi1=args.truth_phi1,
        domain_max=args.domain_max,
        domain_buffer=args.domain_buffer,
        base_seed=args.base_seed,
        output_mat=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    main()
