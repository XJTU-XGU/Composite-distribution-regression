import numpy as np
import pickle
import pandas as pd

############################################################
# 0) 1D empirical Wasserstein
############################################################

def w_dist_1d_empirical(pred, true_sorted, metric="w2"):
    """
    metric:
      - "w2": sqrt(mean((sort(pred) - sort(true))^2))
      - "w1": mean(|sort(pred) - sort(true)|)
    true_sorted: np.sort(true) (pre-sorted for speed)
    """
    pred = np.asarray(pred, dtype=float).ravel()
    m = min(len(pred), len(true_sorted))
    a = np.sort(pred)[:m]
    b = true_sorted[:m]
    if metric.lower() == "w1":
        return float(np.mean(np.abs(a - b)))
    elif metric.lower() == "w2":
        return float(np.sqrt(np.mean((a - b) ** 2)))
    else:
        raise ValueError("metric must be 'w1' or 'w2'.")


############################################################
# 1) Robust sampling from processed_data.pkl
############################################################

def _get_entry_samples(entry, n_samples, rng):
    """
    Robustly obtain samples for one perturbation entry.
    Accepts either:
      - entry["samples"] / entry["value"] (array-like)
      - entry["kde"] (scipy.stats.gaussian_kde) -> resample
    """
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
        return kde.resample(n_samples,seed=rng).reshape(-1)

    if "value" in entry and entry["value"] is not None:
        arr = np.asarray(entry["value"], dtype=float).ravel()
        idx = rng.choice(len(arr), size=n_samples, replace=True)
        return arr[idx]

    raise KeyError("Cannot find 'kde' or samples/value in entry.")


def build_data_from_results(results, n_samples=1000, seed=0):
    """
    Build:
      X_ref: control samples
      Y_all: (N, n_samples) each perturbation samples
      C_all: (N, d) embedding vectors (composite uses sum of two singles)
      pert_names: list length N
      is_composite: bool array length N
    """
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
    xs = np.sort(X)
    ys = np.sort(Y)
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
# 3) Method-1: Ours S(x,c)
############################################################

def fit_global_S_with_c(X_list, Y_list, C_list, m_bins=50, max_iter=50):
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
        X, Y = np.asarray(X_list[i]), np.asarray(Y_list[i])
        Q = empirical_ot_map_1d(X, Y)
        y_ij[i] = Q(centers)
        counts, _ = np.histogram(X, bins=edges)
        w_ij[i] = counts / max(1, len(X))

    eps = 1e-12
    w0 = np.zeros(d, dtype=float)
    w1 = np.zeros(d, dtype=float)
    b0 = 1.0
    b1 = 0.0

    for _ in range(max_iter):
        phi0 = C @ w0 + b0
        phi1 = C @ w1 + b1

        num = (w_ij * phi0[:, None] * (y_ij - phi1[:, None])).sum(axis=0)
        den = (w_ij * (phi0[:, None] ** 2)).sum(axis=0) + eps
        y_bar = num / den
        z = weighted_pava(y_bar, den, eps=eps)

        ZN = np.repeat(z.reshape(1, -1), N, axis=0)
        Y_flat = y_ij.reshape(-1)
        W_flat = w_ij.reshape(-1)
        C_tile = np.repeat(C, m_bins, axis=0)
        Z_expanded = ZN.reshape(-1)
        ones = np.ones_like(Z_expanded)

        X_design = np.column_stack([
            C_tile * Z_expanded[:, None],   # w0
            Z_expanded,                     # b0
            C_tile,                         # w1
            ones                            # b1
        ])

        ok = W_flat > 0
        X_sel = X_design[ok] * np.sqrt(W_flat[ok])[:, None]
        y_sel = Y_flat[ok] * np.sqrt(W_flat[ok])

        beta, *_ = np.linalg.lstsq(X_sel, y_sel, rcond=None)
        w0 = beta[:d]
        b0 = beta[d]
        w1 = beta[d+1:d+1+d]
        b1 = beta[-1]

    def S_hat(x, c_vec):
        x = np.asarray(x, dtype=float)
        c_vec = np.asarray(c_vec, dtype=float)
        base = np.interp(x, centers, z, left=z[0], right=z[-1])
        phi0 = float(c_vec @ w0 + b0)
        phi1 = float(c_vec @ w1 + b1)
        return phi0 * base + phi1

    return centers, z, S_hat


############################################################
# 4) Method-2: Baseline T(x) (no c)
############################################################

def fit_global_T_no_c(X_list, Y_list, m_bins=50):
    N = len(X_list)
    X_all = np.concatenate(X_list)
    xmin, xmax = float(X_all.min()), float(X_all.max())
    edges = np.linspace(xmin, xmax, m_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    w_ij = np.zeros((N, m_bins), dtype=float)
    y_ij = np.zeros((N, m_bins), dtype=float)

    for i in range(N):
        X, Y = np.asarray(X_list[i]), np.asarray(Y_list[i])
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
# 5) Method-3: Mean/Var Gaussian baseline
############################################################

def ridge_fit(Phi, y, lam=1e-2, penalize_intercept=False):
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float)
    XtX = Phi.T @ Phi
    Xty = Phi.T @ y
    P = np.eye(Phi.shape[1], dtype=float)
    if not penalize_intercept:
        P[0, 0] = 0.0
    beta = np.linalg.solve(XtX + lam * P, Xty)
    return beta


def fit_meanvar_gaussian_baseline(X_list, Y_list, C_list, lam=1e-2):
    """
    Features: [1, c(d), mx, vx, c*mx(d), c*vx(d)]
    Targets: mean(Y), log var(Y)
    """
    X_list = [np.asarray(x, dtype=float) for x in X_list]
    Y_list = [np.asarray(y, dtype=float) for y in Y_list]
    C = np.asarray(C_list, dtype=float)
    d = C.shape[1]

    mx = np.array([np.mean(x) for x in X_list], dtype=float)
    vx = np.array([np.var(x, ddof=1) if len(x) > 1 else 0.0 for x in X_list], dtype=float)

    my = np.array([np.mean(y) for y in Y_list], dtype=float)
    vy = np.array([np.var(y, ddof=1) if len(y) > 1 else 1e-8 for y in Y_list], dtype=float)
    logvy = np.log(vy + 1e-8)

    Phi = np.concatenate([
        np.ones((len(X_list), 1), dtype=float),
        C,
        mx.reshape(-1, 1),
        vx.reshape(-1, 1),
        C * mx[:, None],
        C * vx[:, None],
    ], axis=1)

    beta_mean = ridge_fit(Phi, my, lam=lam, penalize_intercept=False)
    beta_logv = ridge_fit(Phi, logvy, lam=lam, penalize_intercept=False)

    def predict_mean_var(mx_t, vx_t, c_t):
        c_t = np.asarray(c_t, dtype=float).ravel()
        feat = np.concatenate([
            np.array([1.0], dtype=float),
            c_t,
            np.array([mx_t, vx_t], dtype=float),
            c_t * mx_t,
            c_t * vx_t,
        ])
        m_pred = float(feat @ beta_mean)
        logv_pred = float(feat @ beta_logv)
        v_pred = float(np.exp(logv_pred))
        return m_pred, max(v_pred, 1e-8)

    return predict_mean_var


############################################################
# 6) LOOCV over composite perturbations + Excel export
############################################################

def summarize_series(x):
    x = np.asarray(x, dtype=float)
    return {
        "Min": float(np.min(x)),
        "Q0.25": float(np.quantile(x, 0.25)),
        "Mean": float(np.mean(x)),
        "Median": float(np.median(x)),
        "Q0.75": float(np.quantile(x, 0.75)),
        "Max": float(np.max(x)),
    }


def run_loocv_over_composites_and_save_excel(
    results,
    out_xlsx="loocv_wasserstein_results.xlsx",
    n_samples=1000,     # smoothed distribution representation size
    n_obs=1000,         # training sample size per perturbation pair
    n_eval=1000,        # evaluation sample size
    m_bins=50,
    max_iter=50,
    lam_b2=1e-2,
    metric="w2",
    seed=2025,
):
    # Build arrays
    X_ref, Y_all, C_all, pert_names, is_comp = build_data_from_results(
        results, n_samples=n_samples, seed=seed
    )
    N = len(pert_names)
    comp_idx = np.where(is_comp)[0]
    if len(comp_idx) == 0:
        raise ValueError("No composite perturbations found for LOOCV.")

    methods = ["Ours", "Cond-free", "Global"]

    # Pre-generate fixed train samples per perturbation (reproducible)
    X_train, Y_train = [], []
    for i in range(N):
        rng_i = np.random.default_rng(seed + 1000 + i)
        X_train.append(rng_i.choice(X_ref, size=n_obs, replace=True))
        Y_train.append(rng_i.choice(Y_all[i], size=n_obs, replace=True))

    # Fixed X_eval and pre-sorted Y_true for stable comparison
    rng_eval = np.random.default_rng(seed + 9999)
    X_eval = rng_eval.choice(X_ref, size=n_eval, replace=True)
    mx_eval = float(np.mean(X_eval))
    vx_eval = float(np.var(X_eval, ddof=1)) if len(X_eval) > 1 else 0.0

    Y_true_sorted = []
    for i in range(N):
        rng_i = np.random.default_rng(seed + 20000 + i)
        y = rng_i.choice(Y_all[i], size=n_eval, replace=True)
        Y_true_sorted.append(np.sort(y))

    all_ids = np.arange(N)

    # Long-form storage for ALL distances (requested)
    # columns: fold, heldout_pert, split, method, pert_name, pert_idx, w
    rows_long = []

    # Per-fold metrics (wide): train_mean and test per method
    rows_fold = []

    for fold, test_id in enumerate(comp_idx):
        train_ids = all_ids[all_ids != test_id]

        X_list = [X_train[i] for i in train_ids]
        Y_list = [Y_train[i] for i in train_ids]
        C_list = [C_all[i] for i in train_ids]

        # Fit models
        _, _, S_hat = fit_global_S_with_c(X_list, Y_list, C_list, m_bins=m_bins, max_iter=max_iter)
        _, _, T_hat = fit_global_T_no_c(X_list, Y_list, m_bins=m_bins)
        pred_mean_var = fit_meanvar_gaussian_baseline(X_list, Y_list, C_list, lam=lam_b2)

        heldout_name = pert_names[test_id]

        # Collect distances for TRAIN per perturbation
        dist_train = {m: [] for m in methods}

        for idx in train_ids:
            true_sorted = Y_true_sorted[idx]
            pert_name = pert_names[idx]

            # Ours
            y_pred = S_hat(X_eval, C_all[idx])
            w = w_dist_1d_empirical(y_pred, true_sorted, metric=metric)
            dist_train["Ours_S(x,c)"].append(w)
            rows_long.append([fold, heldout_name, "train", "Ours_S(x,c)", pert_name, int(idx), float(w)])

            # T(x)
            y_pred = T_hat(X_eval)
            w = w_dist_1d_empirical(y_pred, true_sorted, metric=metric)
            dist_train["Baseline_T(x)"].append(w)
            rows_long.append([fold, heldout_name, "train", "Baseline_T(x)", pert_name, int(idx), float(w)])

            # mean/var Gaussian
            rng_pred = np.random.default_rng(seed + 50000 + fold * 1000 + idx)
            m_pred, v_pred = pred_mean_var(mx_eval, vx_eval, C_all[idx])
            y_pred = rng_pred.normal(loc=m_pred, scale=np.sqrt(v_pred), size=n_eval)
            w = w_dist_1d_empirical(y_pred, true_sorted, metric=metric)
            dist_train["Baseline_meanvar"].append(w)
            rows_long.append([fold, heldout_name, "train", "Baseline_meanvar", pert_name, int(idx), float(w)])

        # TEST distance (held-out composite)
        dist_test = {}

        true_sorted = Y_true_sorted[test_id]
        pert_name = pert_names[test_id]

        y_pred = S_hat(X_eval, C_all[test_id])
        w = w_dist_1d_empirical(y_pred, true_sorted, metric=metric)
        dist_test["Ours_S(x,c)"] = w
        rows_long.append([fold, heldout_name, "test", "Ours_S(x,c)", pert_name, int(test_id), float(w)])

        y_pred = T_hat(X_eval)
        w = w_dist_1d_empirical(y_pred, true_sorted, metric=metric)
        dist_test["Baseline_T(x)"] = w
        rows_long.append([fold, heldout_name, "test", "Baseline_T(x)", pert_name, int(test_id), float(w)])

        rng_pred = np.random.default_rng(seed + 60000 + fold * 1000 + test_id)
        m_pred, v_pred = pred_mean_var(mx_eval, vx_eval, C_all[test_id])
        y_pred = rng_pred.normal(loc=m_pred, scale=np.sqrt(v_pred), size=n_eval)
        w = w_dist_1d_empirical(y_pred, true_sorted, metric=metric)
        dist_test["Baseline_meanvar"] = w
        rows_long.append([fold, heldout_name, "test", "Baseline_meanvar", pert_name, int(test_id), float(w)])

        # Per-fold wide row: train_mean and test
        row = {
            "fold": fold,
            "heldout_pert": heldout_name,
        }
        for m in methods:
            row[f"{m}_train_mean"] = float(np.mean(dist_train[m]))
            row[f"{m}_test"] = float(dist_test[m])
        rows_fold.append(row)

        if (fold + 1) % 10 == 0 or (fold + 1) == len(comp_idx):
            print(f"[{fold+1}/{len(comp_idx)}] done (held-out = {heldout_name})")

    # Build DataFrames
    df_long = pd.DataFrame(
        rows_long,
        columns=["fold", "heldout_pert", "split", "method", "pert_name", "pert_idx", "w"]
    )
    df_fold = pd.DataFrame(rows_fold)

    # Summary tables (over folds)
    # Train summary uses per-fold train_mean; Test summary uses per-fold test distance.
    summary_train_rows = []
    summary_test_rows = []
    for m in methods:
        train_series = df_fold[f"{m}_train_mean"].values
        test_series = df_fold[f"{m}_test"].values
        st_tr = summarize_series(train_series)
        st_te = summarize_series(test_series)
        summary_train_rows.append({"Method": m, **st_tr})
        summary_test_rows.append({"Method": m, **st_te})

    df_summary_train = pd.DataFrame(summary_train_rows)
    df_summary_test = pd.DataFrame(summary_test_rows)

    # Pretty print in console
    def _print_table(title, df_sum):
        cols = ["Method", "Min", "Q0.25", "Mean", "Median", "Q0.75", "Max"]
        print("\n" + "=" * 90)
        print(f"{title} | metric = {metric.upper()} | folds = {len(comp_idx)}")
        print(df_sum[cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
        print("=" * 90 + "\n")

    _print_table("SUMMARY (TRAIN: per-fold mean over training perturbations)", df_summary_train)
    _print_table("SUMMARY (TEST: held-out composite per fold)", df_summary_test)

    # Save to Excel
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_summary_train.to_excel(writer, sheet_name="summary_train", index=False)
        df_summary_test.to_excel(writer, sheet_name="summary_test", index=False)
        df_fold.to_excel(writer, sheet_name="fold_metrics", index=False)
        df_long.to_excel(writer, sheet_name="all_distances", index=False)

    print(f"[Saved] {out_xlsx}")

    return {
        "df_summary_train": df_summary_train,
        "df_summary_test": df_summary_test,
        "df_fold": df_fold,
        "df_long": df_long,
        "out_xlsx": out_xlsx,
        "metric": metric,
    }


############################################################
# 7) MAIN
############################################################

if __name__ == "__main__":
    with open("processed_data.pkl", "rb") as f:
        results = pickle.load(f)

    res = run_loocv_over_composites_and_save_excel(
        results,
        out_xlsx="loocv_wasserstein_results.xlsx",
        n_samples=1000,
        n_obs=1000,
        n_eval=1000,
        m_bins=50,
        max_iter=50,
        lam_b2=1e-2,
        metric="w2",   # or "w1"
        seed=2026,
    )