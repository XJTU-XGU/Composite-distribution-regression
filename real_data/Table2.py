import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))

try:
    from partially_global_frechet import fit_pgfr_baseline_from_samples
except ImportError as e:
    raise ImportError(
        "Cannot import fit_pgfr_baseline_from_samples from partially_global_frechet.py. "
        "Please make sure that file is in the same directory as this script."
    ) from e


OUTPUT_DIR = "outputs"


def _resolve_output_path(path):
    if not os.path.dirname(path):
        path = os.path.join(OUTPUT_DIR, path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


############################################################
# 0) 1D empirical Wasserstein
############################################################
def w_dist_1d_empirical(pred, true_sorted, metric="w2"):
    pred = np.asarray(pred, dtype=float).ravel()
    true_sorted = np.asarray(true_sorted, dtype=float).ravel()
    m = min(len(pred), len(true_sorted))
    a = np.sort(pred)[:m]
    b = true_sorted[:m]

    if metric.lower() == "w1":
        return float(np.mean(np.abs(a - b)))
    if metric.lower() == "w2":
        return float(np.sqrt(np.mean((a - b) ** 2)))
    raise ValueError("metric must be 'w1' or 'w2'.")


############################################################
# 1) Robust sampling from processed_data.pkl
############################################################
def _get_entry_samples(entry, n_samples, rng):
    if entry is None:
        raise ValueError("Empty entry")

    if "samples" in entry and entry["samples"] is not None:
        arr = np.asarray(entry["samples"], dtype=float).ravel()
        idx = rng.choice(len(arr), size=n_samples, replace=True)
        return arr[idx]

    if "value" in entry and entry["value"] is not None and ("kde" not in entry or entry["kde"] is None):
        arr = np.asarray(entry["value"], dtype=float).ravel()
        idx = rng.choice(len(arr), size=n_samples, replace=True)
        return arr[idx]

    if "kde" in entry and entry["kde"] is not None:
        kde = entry["kde"]
        return kde.resample(n_samples, seed=rng).reshape(-1)

    if "value" in entry and entry["value"] is not None:
        arr = np.asarray(entry["value"], dtype=float).ravel()
        idx = rng.choice(len(arr), size=n_samples, replace=True)
        return arr[idx]

    raise KeyError("Cannot find 'kde' or 'samples'/'value' in entry.")


def build_data_from_results(results, n_samples=1000, seed=0):
    rng = np.random.default_rng(seed)
    perts = sorted(results.keys())
    single_perts, composite_perts = [], []

    for p in perts:
        if p == "control":
            continue
        rep = results[p].get("representation", None)
        if rep is None:
            continue
        if isinstance(rep, tuple):
            composite_perts.append(p)
        else:
            single_perts.append(p)

    X_ref = _get_entry_samples(results["control"], n_samples, rng)

    Y_all, C_all, pert_names, is_composite = [], [], [], []
    for p in single_perts + composite_perts:
        info = results[p]
        Y_i = _get_entry_samples(info, n_samples, rng)

        rep = info["representation"]
        if isinstance(rep, tuple):
            c_i = np.asarray(rep[0], dtype=float) + np.asarray(rep[1], dtype=float)
            comp = True
        else:
            c_i = np.asarray(rep, dtype=float)
            comp = False

        Y_all.append(Y_i)
        C_all.append(c_i)
        pert_names.append(p)
        is_composite.append(comp)

    return (
        np.asarray(X_ref, dtype=float),
        np.asarray(Y_all, dtype=float),
        np.asarray(C_all, dtype=float),
        pert_names,
        np.asarray(is_composite, dtype=bool),
    )


############################################################
# 2) OT map + Weighted PAVA
############################################################
def empirical_ot_map_1d(X, Y):
    xs = np.sort(np.asarray(X, dtype=float))
    ys = np.sort(np.asarray(Y, dtype=float))

    def Q(xq):
        return np.interp(xq, xs, ys, left=ys[0], right=ys[-1])

    return Q


def weighted_pava(y, w, eps=1e-12):
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float) + eps
    levels, weights, sizes = [], [], []

    for j in range(len(y)):
        levels.append(y[j])
        weights.append(w[j])
        sizes.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            y2, y1 = levels.pop(), levels.pop()
            w2, w1 = weights.pop(), weights.pop()
            s2, s1 = sizes.pop(), sizes.pop()
            w_new = w1 + w2
            lvl_new = (y1 * w1 + y2 * w2) / w_new
            levels.append(lvl_new)
            weights.append(w_new)
            sizes.append(s1 + s2)

    z = []
    for lvl, sz in zip(levels, sizes):
        z.extend([lvl] * sz)
    return np.asarray(z, dtype=float)


############################################################
# 3) Method-1: CSOT S(x,c) for vector covariates
#    Added tol-based early stopping
############################################################
def fit_global_S_with_c(X_list, Y_list, C_list, m_bins=50, max_iter=50, tol=1e-5):
    N = len(X_list)
    C = np.asarray(C_list, dtype=float)
    d = C.shape[1]

    X_all = np.concatenate(X_list)
    xmin, xmax = float(X_all.min()), float(X_all.max())
    edges = np.linspace(xmin, xmax, m_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    w_ij = np.zeros((N, m_bins), dtype=float)
    y_ij = np.zeros((N, m_bins), dtype=float)

    for i in range(N):
        X = np.asarray(X_list[i], dtype=float)
        Y = np.asarray(Y_list[i], dtype=float)
        Q = empirical_ot_map_1d(X, Y)
        y_ij[i] = Q(centers)
        counts, _ = np.histogram(X, bins=edges)
        w_ij[i] = counts / max(1, len(X))

    eps = 1e-12

    # initialization
    w0 = np.zeros(d, dtype=float)
    w1 = np.zeros(d, dtype=float)
    b0 = 1.0
    b1 = 0.0

    # initial z under phi0(c)=1, phi1(c)=0
    phi0 = C @ w0 + b0
    phi1 = C @ w1 + b1
    phi0_mat = phi0[:, None]
    phi1_mat = phi1[:, None]

    num0 = (w_ij * phi0_mat * (y_ij - phi1_mat)).sum(axis=0)
    den0 = (w_ij * (phi0_mat ** 2)).sum(axis=0) + eps
    y_bar0 = num0 / den0
    z = weighted_pava(y_bar0, den0, eps=eps)

    for _ in range(max_iter):
        z_old = z.copy()
        w0_old = w0.copy()
        w1_old = w1.copy()
        b0_old = b0
        b1_old = b1

        # Step 1: update z by weighted PAVA
        phi0 = C @ w0 + b0
        phi1 = C @ w1 + b1
        phi0_mat = phi0[:, None]
        phi1_mat = phi1[:, None]

        num = (w_ij * phi0_mat * (y_ij - phi1_mat)).sum(axis=0)
        den = (w_ij * (phi0_mat ** 2)).sum(axis=0) + eps
        y_bar = num / den
        z = weighted_pava(y_bar, den, eps=eps)

        # Step 2: update (w0,b0,w1,b1) by weighted LS
        ZN = np.repeat(z.reshape(1, -1), N, axis=0)
        Y_flat = y_ij.reshape(-1)
        W_flat = w_ij.reshape(-1)
        C_tile = np.repeat(C, m_bins, axis=0)
        Z_expanded = ZN.reshape(-1)
        ones = np.ones_like(Z_expanded)

        X_design = np.column_stack([
            C_tile * Z_expanded[:, None],  # w0 part
            Z_expanded,                    # b0 part
            C_tile,                        # w1 part
            ones,                          # b1 part
        ])

        ok = W_flat > 0
        if ok.sum() > 0:
            X_sel = X_design[ok] * np.sqrt(W_flat[ok])[:, None]
            y_sel = Y_flat[ok] * np.sqrt(W_flat[ok])

            beta, *_ = np.linalg.lstsq(X_sel, y_sel, rcond=None)
            w0 = beta[:d]
            b0 = beta[d]
            w1 = beta[d + 1:d + 1 + d]
            b1 = beta[-1]

        # convergence check
        delta_z = np.max(np.abs(z - z_old))
        delta_params = max(
            np.max(np.abs(w0 - w0_old)),
            abs(b0 - b0_old),
            np.max(np.abs(w1 - w1_old)),
            abs(b1 - b1_old),
        )

        if max(delta_z, delta_params) < tol:

            break

    def S_hat(x, c_vec):
        x = np.asarray(x, dtype=float)
        c_vec = np.asarray(c_vec, dtype=float).ravel()
        base = np.interp(x, centers, z, left=z[0], right=z[-1])
        phi0 = float(c_vec @ w0 + b0)
        phi1 = float(c_vec @ w1 + b1)
        return phi0 * base + phi1

    return centers, z, S_hat


############################################################
# 4) Method-2: OTM T(x) (no c)
############################################################
def fit_otm_no_c(X_list, Y_list, m_bins=50):
    N = len(X_list)
    X_all = np.concatenate(X_list)
    xmin, xmax = float(X_all.min()), float(X_all.max())
    edges = np.linspace(xmin, xmax, m_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    w_ij = np.zeros((N, m_bins), dtype=float)
    y_ij = np.zeros((N, m_bins), dtype=float)

    for i in range(N):
        X = np.asarray(X_list[i], dtype=float)
        Y = np.asarray(Y_list[i], dtype=float)
        Q = empirical_ot_map_1d(X, Y)
        y_ij[i] = Q(centers)
        counts, _ = np.histogram(X, bins=edges)
        w_ij[i] = counts / max(1, len(X))

    eps = 1e-12
    den = w_ij.sum(axis=0) + eps
    num = (w_ij * y_ij).sum(axis=0)
    y_bar = num / den
    z = weighted_pava(y_bar, den, eps=eps)

    def T_hat(x):
        x = np.asarray(x, dtype=float)
        return np.interp(x, centers, z, left=z[0], right=z[-1])

    return centers, z, T_hat


############################################################
# 5) Summary helper
############################################################
def summarize_series(x):
    x = np.asarray(x, dtype=float)
    return {
        "Min": float(np.min(x)),
        "Q0.25": float(np.quantile(x, 0.25)),
        "Median": float(np.median(x)),
        "Q0.75": float(np.quantile(x, 0.75)),
        "Max": float(np.max(x)),
        "Mean": float(np.mean(x)),
    }


############################################################
# 6) LOOCV over composite perturbations
#    Only held-out test summary + one Time column
############################################################
def run_loocv_over_composites_test_only(
    results,
    out_xlsx="loocv_test_summary_with_time.xlsx",
    n_samples=1000,
    n_obs=1000,
    n_eval=1000,
    m_bins=50,
    max_iter=50,
    tol=1e-5,
    metric="w2",
    seed=2025,
    pgfr_bandwidth=None,
    pgfr_bandwidth_scale=2.0,
    pgfr_n_quantiles=200,
):
    X_ref, Y_all, C_all, pert_names, is_comp = build_data_from_results(
        results, n_samples=n_samples, seed=seed
    )
    N = len(pert_names)
    comp_idx = np.where(is_comp)[0]
    if len(comp_idx) == 0:
        raise ValueError("No composite perturbations found for LOOCV.")

    methods = ["CSOT", "OTM", "PGFR"]

    X_train, Y_train = [], []
    for i in range(N):
        rng_i = np.random.default_rng(seed + 1000 + i)
        X_train.append(rng_i.choice(X_ref, size=n_obs, replace=True))
        Y_train.append(rng_i.choice(Y_all[i], size=n_obs, replace=True))

    rng_eval = np.random.default_rng(seed + 9999)
    X_eval = rng_eval.choice(X_ref, size=n_eval, replace=True)

    Y_true_sorted = []
    for i in range(N):
        rng_i = np.random.default_rng(seed + 20000 + i)
        y = rng_i.choice(Y_all[i], size=n_eval, replace=True)
        Y_true_sorted.append(np.sort(y))

    all_ids = np.arange(N)
    rows_fold = []
    total_time = {m: 0.0 for m in methods}

    for fold, test_id in enumerate(comp_idx):
        train_ids = all_ids[all_ids != test_id]

        X_list = [X_train[i] for i in train_ids]
        Y_list = [Y_train[i] for i in train_ids]
        C_list = [C_all[i] for i in train_ids]

        heldout_name = pert_names[test_id]
        true_sorted = Y_true_sorted[test_id]
        c_vec = C_all[test_id]

        row = {
            "fold": fold,
            "heldout_pert": heldout_name,
        }

        # CSOT
        t0 = time.perf_counter()
        _, _, S_hat = fit_global_S_with_c(
            X_list, Y_list, C_list, m_bins=m_bins, max_iter=max_iter, tol=tol
        )
        y_pred = S_hat(X_eval, c_vec)
        w_csot = w_dist_1d_empirical(y_pred, true_sorted, metric=metric)
        t_csot = time.perf_counter() - t0
        total_time["CSOT"] += t_csot

        row["CSOT_test"] = float(w_csot)
        row["CSOT_time"] = float(t_csot)

        # OTM
        t0 = time.perf_counter()
        _, _, T_hat = fit_otm_no_c(X_list, Y_list, m_bins=m_bins)
        y_pred = T_hat(X_eval)
        w_otm = w_dist_1d_empirical(y_pred, true_sorted, metric=metric)
        t_otm = time.perf_counter() - t0
        total_time["OTM"] += t_otm

        row["OTM_test"] = float(w_otm)
        row["OTM_time"] = float(t_otm)

        # PGFR
        t0 = time.perf_counter()
        pgfr_predictor, pgfr_info = fit_pgfr_baseline_from_samples(
            X_list=X_list,
            Y_list=Y_list,
            c_list=C_list,
            bandwidth=pgfr_bandwidth,
            bandwidth_scale=pgfr_bandwidth_scale,
            n_quantiles=pgfr_n_quantiles,
        )
        y_pred = pgfr_predictor(X_eval, c_vec, n_pred=n_eval)
        w_pgfr = w_dist_1d_empirical(y_pred, true_sorted, metric=metric)
        t_pgfr = time.perf_counter() - t0
        total_time["PGFR"] += t_pgfr

        row["PGFR_test"] = float(w_pgfr)
        row["PGFR_time"] = float(t_pgfr)
        row["pgfr_bandwidth"] = float(pgfr_info["bandwidth"])

        rows_fold.append(row)

        print(
            f"[{fold + 1}/{len(comp_idx)}] done (held-out = {heldout_name}) | "
            f"CSOT: w={w_csot:.4f}, t={t_csot:.4f}s | "
            f"OTM: w={w_otm:.4f}, t={t_otm:.4f}s | "
            f"PGFR: w={w_pgfr:.4f}, t={t_pgfr:.4f}s"
        )

    df_fold = pd.DataFrame(rows_fold)

    summary_rows = []
    for m, display_name in [
        ("OTM", " Ghodrati & Panaretos (2022)"),
        ("PGFR", "Tucker & Wu (2025)"),
        ("CSOT", "Our method"),
    ]:
        stats = summarize_series(df_fold[f"{m}_test"].values)
        stats["Time"] = float(total_time[m])
        summary_rows.append({"Method": display_name, **stats})

    df_summary_test = pd.DataFrame(summary_rows)

    cols = ["Method", "Min", "Q0.25", "Median", "Q0.75", "Max", "Mean", "Time"]

    print("\n" + "=" * 110)
    print(f"TEST SUMMARY | metric = {metric.upper()} | folds = {len(comp_idx)}")
    print(df_summary_test[cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print("=" * 110 + "\n")

    out_xlsx = _resolve_output_path(out_xlsx)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_summary_test.to_excel(writer, sheet_name="summary_test", index=False)
        df_fold.to_excel(writer, sheet_name="fold_metrics", index=False)

    print(f"[Saved Excel] {out_xlsx}")

    return {
        "df_summary_test": df_summary_test,
        "df_fold": df_fold,
        "out_xlsx": out_xlsx,
        "metric": metric,
        "total_time": total_time,
    }


############################################################
# 7) MAIN
############################################################
if __name__ == "__main__":
    with open("processed_data.pkl", "rb") as f:
        results = pickle.load(f)

    run_loocv_over_composites_test_only(
        results,
        out_xlsx="loocv_test_summary_with_time.xlsx",
        n_samples=1000,
        n_obs=1000,
        n_eval=1000,
        m_bins=50,
        max_iter=50,
        tol=1e-5,
        metric="w2",   # or "w1"
        seed=2026,
        pgfr_bandwidth=None,
        pgfr_bandwidth_scale=1,
        pgfr_n_quantiles=200,
    )
