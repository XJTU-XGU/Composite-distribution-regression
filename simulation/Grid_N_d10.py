#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the fig3 vector-covariate no-intercept simulation with fixed p=11
(d=p-1=10) while varying the training sample size N.

The data-generation and estimator implementations are reused from
Grid_d_N100.py.
GOT_noisy_scalar is intentionally excluded because it is too slow for this
N-grid experiment.

Saved result format matches the existing torch-saved .mat files:

    results["N_10"][method] = {"w1": [...], "time": total_seconds}

where the default methods are structured_transport, GP2022_condfree_OT,
and TW2025_PGFR.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import Grid_d_N100 as base

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

METHOD_NAMES = (
    "structured_transport",
    "GP2022_condfree_OT",
    "TW2025_PGFR",
)


def make_methods_without_got(
    m_fit: int = 120,
    m_bins: int = 80,
    tw_bandwidth: Optional[float] = None,
    tw_bandwidth_scale: float = 1.0,
    tw_pred_sample_size: Optional[int] = None,
):
    """Construct the three retained methods for one Monte Carlo replicate."""
    return {
        "structured_transport": base.Fig4StructuredTransportRegressor(
            m_bins=m_bins,
            max_iter=80,
            tol=1e-6,
        ),
        "GP2022_condfree_OT": base.Fig4GP2022IgnoreScalarRegressor(
            m_bins=m_bins,
        ),
        "TW2025_PGFR": base.ExternalTuckerWuPGFRRegressor(
            n_quantiles=m_fit,
            bandwidth=tw_bandwidth,
            bandwidth_scale=tw_bandwidth_scale,
            quantile_eps=1e-4,
            pred_sample_size=tw_pred_sample_size,
        ),
    }


def run_one_replicate(
    p: int,
    rep: int,
    seed: int,
    N_train: int,
    N_test: int = 1000,
    n_obs: int = 100,
    m_fit: int = 120,
    m_bins: int = 80,
    pis: Tuple[float, float, float] = (0.3, 0.4, 0.3),
    J_segments: int = 8,
    scalar_range: Tuple[float, float] = (0.0, 1.5),
    scalar_scaling: str = "none",
    phi0_l2_sq: float = 1.0,
    phi1_l2_sq: float = 0.25,
    theta_decay0: float = 2.0,
    theta_decay1: float = 1.0,
    domain_max: float = 4.0,
    domain_buffer: float = 0.25,
    tw_bandwidth: Optional[float] = None,
    tw_bandwidth_scale: float = 1.0,
    tw_pred_sample_size: Optional[int] = None,
    verbose: bool = False,
) -> List[Dict[str, object]]:
    """Generate one data split and evaluate the retained methods."""
    data = base.generate_fig3_vector_style_data(
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

    methods = make_methods_without_got(
        m_fit=m_fit,
        m_bins=m_bins,
        tw_bandwidth=tw_bandwidth,
        tw_bandwidth_scale=tw_bandwidth_scale,
        tw_pred_sample_size=tw_pred_sample_size,
    )
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
                f"[WARNING] {method_name} failed for N={N_train}, "
                f"p={p}, rep={rep}: {exc}",
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


def run_N_grid_experiment(
    N_list: Sequence[int] = (10, 100, 1000),
    p: int = 11,
    reps: int = 100,
    N_test: int = 1000,
    n_obs: int = 100,
    m_fit: int = 120,
    m_bins: int = 80,
    pis: Tuple[float, float, float] = (0.3, 0.4, 0.3),
    J_segments: int = 8,
    scalar_range: Tuple[float, float] = (0.0, 1.5),
    scalar_scaling: str = "none",
    phi0_l2_sq: float = 1.0,
    phi1_l2_sq: float = 0.25,
    theta_decay0: float = 2.0,
    theta_decay1: float = 1.0,
    domain_max: float = 4.0,
    domain_buffer: float = 0.25,
    base_seed: int = 2026,
    tw_bandwidth: Optional[float] = None,
    tw_bandwidth_scale: float = 1.0,
    tw_pred_sample_size: Optional[int] = None,
    output_mat: str = "results/fig3_vector_decay_theta_no_intercept_N_grid_p11_results.mat",
    verbose: bool = True,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Run fixed-p N-grid experiment and save W1 lists plus total times."""
    results: Dict[str, Dict[str, Dict[str, object]]] = {}
    output_path = Path(output_mat)

    for N_train in N_list:
        N_key = f"N_{int(N_train)}"
        N_results = results.setdefault(N_key, {})
        for method_name in METHOD_NAMES:
            N_results.setdefault(method_name, {"w1": [], "time": 0.0})

        for rep in range(int(reps)):
            seed = int(base_seed + 100000 * int(p) + rep)
            if verbose:
                print(
                    f"[N={N_train}, p={p}, d={p - 1}, rep={rep + 1}/{reps}, "
                    f"seed={seed}]",
                    flush=True,
                )

            run_rows = run_one_replicate(
                p=int(p),
                rep=int(rep),
                seed=seed,
                N_train=int(N_train),
                N_test=int(N_test),
                n_obs=int(n_obs),
                m_fit=int(m_fit),
                m_bins=int(m_bins),
                pis=pis,
                J_segments=int(J_segments),
                scalar_range=scalar_range,
                scalar_scaling=scalar_scaling,
                phi0_l2_sq=float(phi0_l2_sq),
                phi1_l2_sq=float(phi1_l2_sq),
                theta_decay0=float(theta_decay0),
                theta_decay1=float(theta_decay1),
                domain_max=float(domain_max),
                domain_buffer=float(domain_buffer),
                tw_bandwidth=tw_bandwidth,
                tw_bandwidth_scale=float(tw_bandwidth_scale),
                tw_pred_sample_size=tw_pred_sample_size,
                verbose=verbose,
            )

            for row in run_rows:
                if row["status"] != "ok":
                    continue
                method_name = str(row["method"])
                method_result = N_results.setdefault(method_name, {"w1": [], "time": 0.0})
                method_result["w1"].append(float(row["w1"]))
                method_result["time"] += float(row["total_train_test_time"])

            # Save after every replicate so a long run can be resumed manually if needed.
            save_results(results, str(output_path))

    final_path = save_results(results, str(output_path))
    if verbose:
        print(f"\nSaved results to: {final_path.resolve()}")
        print_time_summary(results)
    return results


def print_time_summary(results: Dict[str, Dict[str, Dict[str, object]]]) -> None:
    print("\nTotal times in seconds:")
    header = ["method"] + list(results.keys())
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for method_name in METHOD_NAMES:
        row = [method_name]
        for N_key in results:
            value = results[N_key].get(method_name, {}).get("time", "")
            row.append("" if value == "" else f"{float(value):.6g}")
        print(" | ".join(row))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fixed p=11 fig3 vector-covariate no-intercept simulation over "
            "N=10,100,1000, excluding GOT_noisy_scalar."
        )
    )
    parser.add_argument("--N-list", "--n-list", type=int, nargs="+", default=[25,50,100])
    parser.add_argument("--p", type=int, default=11)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--N-test", "--n-test", type=int, default=1000)
    parser.add_argument("--n-obs", type=int, default=100)
    parser.add_argument("--m-fit", type=int, default=120)
    parser.add_argument("--m-bins", type=int, default=80)
    parser.add_argument("--pis", type=float, nargs=3, default=[0.3, 0.4, 0.3])
    parser.add_argument("--J-segments", type=int, default=8)
    parser.add_argument("--scalar-range", type=float, nargs=2, default=[0.0, 1.5])
    parser.add_argument("--scalar-scaling", type=str, default="none", choices=["none", "sqrt_d"])
    parser.add_argument("--phi0-l2-sq", type=float, default=1.0)
    parser.add_argument("--phi1-l2-sq", type=float, default=0.25)
    parser.add_argument("--theta-decay0", type=float, default=2.0)
    parser.add_argument("--theta-decay1", type=float, default=1.0)
    parser.add_argument("--domain-max", type=float, default=4.0)
    parser.add_argument("--domain-buffer", type=float, default=0.25)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--tw-bandwidth", type=float, default=None)
    parser.add_argument("--tw-bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--tw-pred-sample-size", type=int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="results/fig3_vector_decay_theta_no_intercept_N_grid_p11_results.mat",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    return run_N_grid_experiment(
        N_list=args.N_list,
        p=args.p,
        reps=args.reps,
        N_test=args.N_test,
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
        base_seed=args.base_seed,
        tw_bandwidth=args.tw_bandwidth,
        tw_bandwidth_scale=args.tw_bandwidth_scale,
        tw_pred_sample_size=args.tw_pred_sample_size,
        output_mat=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    main()
