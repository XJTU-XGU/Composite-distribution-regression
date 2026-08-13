import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# make sure local files are importable
sys.path.append(os.path.dirname(__file__))

from Structured_transport_map_core import (
    zeta_k,
    sample_mu_params,
    sample_from_mixture_beta,
    sample_random_S_eps,
    empirical_ot_map_1d,
    weighted_pava,
)
from partially_global_frechet import fit_pgfr_baseline_from_samples


# =========================
# Global style
# =========================
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 18,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


# ============================================================
# Section 4.2: domain convention switch
# ============================================================
FORCE_DOMAIN_0_3 = False
DOMAIN_0_3 = (0.0, 3.0)


# ============================================================
# Utilities
# ============================================================

def beta_mixture_moments(alphas, betas, pis):
    """Exact mean/variance of a Beta mixture."""
    alphas = np.asarray(alphas, dtype=float)
    betas = np.asarray(betas, dtype=float)
    pis = np.asarray(pis, dtype=float)

    m = alphas / (alphas + betas)
    v = (alphas * betas) / ((alphas + betas) ** 2 * (alphas + betas + 1.0))

    second = v + m ** 2
    mean = float(np.sum(pis * m))
    second_moment = float(np.sum(pis * second))
    var = float(max(second_moment - mean ** 2, 1e-12))
    return mean, var



def S0_true_dependent(x, strategy, mean_mu, var_mu, eps=1e-12):
    """
    Dependent configuration base transport map S0.

    - whitening:
        S0(x;c) = (x - mean_mu) / sqrt(var_mu)
      If FORCE_DOMAIN_0_3=True, map standardized values to [0,3].

    - global:
        S0(x;c) = zeta_4(x), independent of mu.
      If FORCE_DOMAIN_0_3=True, use 3*zeta_4(x).
    """
    x = np.asarray(x, dtype=float)

    if strategy == "whitening":
        z = (x - mean_mu) / np.sqrt(var_mu + eps)
        if FORCE_DOMAIN_0_3:
            return 0.75 * z + 1.5
        return z

    if strategy == "global":
        g = zeta_k(x, k=4)
        if FORCE_DOMAIN_0_3:
            return 3.0 * g
        return g

    raise ValueError(f"Unknown strategy: {strategy}")



def get_domain_for_strategy(strategy):
    if FORCE_DOMAIN_0_3:
        return DOMAIN_0_3

    if strategy == "whitening":
        return -3.0, 3.0
    if strategy == "global":
        return 0.0, 1.0
    raise ValueError(f"Unknown strategy: {strategy}")



def w2_1d_empirical(a, b):
    """1D 2-Wasserstein distance between two empirical samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = min(len(a), len(b))
    aa = np.sort(a)[:m]
    bb = np.sort(b)[:m]
    return float(np.sqrt(np.mean((aa - bb) ** 2)))


# ============================================================
# Grid preparation + estimators
# ============================================================

def prepare_grid_data(X_list, Y_list, m_bins=50):
    """Compute shared grid quantities for Cond-free / Ours."""
    N = len(X_list)
    edges = np.linspace(0.0, 1.0, m_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    w_ij = np.zeros((N, m_bins), dtype=float)
    y_ij = np.zeros((N, m_bins), dtype=float)

    for i in range(N):
        X = np.asarray(X_list[i], dtype=float)
        Y = np.asarray(Y_list[i], dtype=float)

        Q_i, _, _ = empirical_ot_map_1d(X, Y)
        y_ij[i, :] = Q_i(centers)

        counts, _ = np.histogram(X, bins=edges)
        w_ij[i, :] = counts.astype(float) / float(len(X))

    return centers, w_ij, y_ij



def fit_condfree_from_grid(centers, w_ij, y_ij):
    """Cond-free baseline: pooled isotonic regression."""
    w_bar = np.sum(w_ij, axis=0) + 1e-12
    y_bar = np.sum(w_ij * y_ij, axis=0) / w_bar
    z_hat = weighted_pava(y_bar, w_bar)

    def S_hat(x):
        x = np.asarray(x, dtype=float)
        return np.interp(x, centers, z_hat, left=z_hat[0], right=z_hat[-1])

    return z_hat, S_hat



def fit_ours_cvec_from_grid(
    centers,
    w_ij,
    y_ij,
    C_list,
    max_iter=50,
    tol=1e-5,
    ridge=1e-8,
    enforce_phi0_positive=True,
):
    """
    Ours with vector-valued condition c in R^d:
        S(x; c) = phi0(c) * z(x) + phi1(c)
    with
        phi0(c) = b0 + a0^T c,
        phi1(c) = b1 + a1^T c.
    """
    C = np.asarray(C_list, dtype=float)
    if C.ndim != 2:
        raise ValueError("C_list must have shape (N, d).")
    N, d = C.shape
    m_bins = len(centers)

    a0 = np.zeros(d, dtype=float)
    b0 = 1.0
    a1 = np.zeros(d, dtype=float)
    b1 = 0.0

    def phi0_vals():
        return b0 + C @ a0

    def phi1_vals():
        return b1 + C @ a1

    # init z by pooled average then project with PAVA
    w_bar = np.sum(w_ij, axis=0) + 1e-12
    y_bar = np.sum(w_ij * y_ij, axis=0) / w_bar
    z = weighted_pava(y_bar, w_bar)

    prev_obj = np.inf
    eps = 1e-12

    for _ in range(max_iter):
        # Step 1: update z via weighted PAVA
        phi0 = phi0_vals()
        phi1 = phi1_vals()
        if enforce_phi0_positive:
            phi0 = np.maximum(phi0, 1e-6)

        phi0_mat = phi0[:, None]
        phi1_mat = phi1[:, None]
        r_ij = (y_ij - phi1_mat) / (phi0_mat + eps)
        w_tilde = w_ij * (phi0_mat ** 2)

        w_bar2 = np.sum(w_tilde, axis=0) + eps
        y_bar2 = np.sum(w_tilde * r_ij, axis=0) / w_bar2
        z_new = weighted_pava(y_bar2, w_bar2)

        # Step 2: update linear parameters via weighted LS
        p = 2 * d + 2
        A = np.zeros((p, p), dtype=float)
        b = np.zeros(p, dtype=float)

        for i in range(N):
            wi = w_ij[i, :]
            yi = y_ij[i, :]
            ci = C[i, :]
            zcol = z_new

            Fi = np.empty((m_bins, p), dtype=float)
            for k in range(d):
                Fi[:, k] = ci[k] * zcol
            Fi[:, d] = zcol
            for k in range(d):
                Fi[:, d + 1 + k] = ci[k]
            Fi[:, -1] = 1.0

            WFi = Fi * wi[:, None]
            A += Fi.T @ WFi
            b += Fi.T @ (wi * yi)

        theta = np.linalg.solve(A + ridge * np.eye(p), b)
        a0 = theta[:d]
        b0 = theta[d]
        a1 = theta[d + 1 : d + 1 + d]
        b1 = theta[-1]

        phi0 = phi0_vals()
        phi1 = phi1_vals()
        if enforce_phi0_positive:
            phi0 = np.maximum(phi0, 1e-6)

        pred = phi0[:, None] * z_new[None, :] + phi1[:, None]
        obj = float(np.sum(w_ij * (y_ij - pred) ** 2))

        if abs(prev_obj - obj) / (prev_obj + 1e-9) < tol:
            z = z_new
            break

        prev_obj = obj
        z = z_new

    def S_hat(x, c_vec):
        x = np.asarray(x, dtype=float)
        c_vec = np.asarray(c_vec, dtype=float).reshape(-1)
        if c_vec.shape[0] != d:
            raise ValueError(f"c_vec must have shape ({d},), got {c_vec.shape}")
        z_x = np.interp(x, centers, z, left=z[0], right=z[-1])
        phi0_c = b0 + float(np.dot(a0, c_vec))
        phi1_c = b1 + float(np.dot(a1, c_vec))
        if enforce_phi0_positive:
            phi0_c = max(phi0_c, 1e-6)
        return phi0_c * z_x + phi1_c

    params = dict(a0=a0, b0=b0, a1=a1, b1=b1, z=z)
    return params, S_hat


# ============================================================
# One replicate
# ============================================================

def run_one_replicate(
    strategy,
    n_obs,
    N_pairs,
    pis=(0.3, 0.4, 0.3),
    J_segments=8,
    m_bins=50,
    num_eval_mu=200,
    num_grid_x=800,
    n_test=1000,
    N_test_pairs=200,
    seed=0,
    pgfr_bandwidth=None,
    pgfr_bandwidth_scale=1.0,
    pgfr_n_quantiles=200,
):
    np.random.seed(seed)
    dom_min, dom_max = get_domain_for_strategy(strategy)

    # --------- generate training data ----------
    X_list, Y_list, C_list = [], [], []
    for _ in range(N_pairs):
        alphas, betas = sample_mu_params()
        mean_mu, var_mu = beta_mixture_moments(alphas, betas, pis)

        X = sample_from_mixture_beta(alphas, betas, pis, size=n_obs)
        c_emp = np.array([np.mean(X), np.var(X)], dtype=float)

        S_eps = sample_random_S_eps(J=J_segments, domain_min=dom_min, domain_max=dom_max)
        Y = S_eps(S0_true_dependent(X, strategy=strategy, mean_mu=mean_mu, var_mu=var_mu))

        X_list.append(X)
        Y_list.append(Y)
        C_list.append(c_emp)

    C_list = np.asarray(C_list, dtype=float)

    centers, w_ij, y_ij = prepare_grid_data(X_list, Y_list, m_bins=m_bins)

    # --------- fit Cond-free / Ours ----------
    _, S_cf = fit_condfree_from_grid(centers, w_ij, y_ij)
    _, S_ours = fit_ours_cvec_from_grid(centers, w_ij, y_ij, C_list)

    # --------- fit PGFR ----------
    pgfr_predictor, pgfr_info = fit_pgfr_baseline_from_samples(
        X_list=X_list,
        Y_list=Y_list,
        c_list=C_list,
        bandwidth=pgfr_bandwidth,
        bandwidth_scale=pgfr_bandwidth_scale,
        n_quantiles=pgfr_n_quantiles,
    )

    # --------- Metric 1: map error ----------
    grid_x = np.linspace(0.0, 1.0, num_grid_x)
    se_ours = []
    se_cf = []
    for _ in range(num_eval_mu):
        alphas, betas = sample_mu_params()
        mean_mu, var_mu = beta_mixture_moments(alphas, betas, pis)
        c_true = np.array([mean_mu, var_mu], dtype=float)

        S0_vals = S0_true_dependent(grid_x, strategy, mean_mu, var_mu)
        se_ours.append(np.mean((S_ours(grid_x, c_true) - S0_vals) ** 2))
        se_cf.append(np.mean((S_cf(grid_x) - S0_vals) ** 2))

    err_L2_ours = float(np.mean(se_ours))
    err_L2_cf = float(np.mean(se_cf))

    # --------- Metric 2: distribution error ----------
    w2_ours = []
    w2_cf = []
    w2_pgfr = []
    for _ in range(N_test_pairs):
        alphas, betas = sample_mu_params()
        mean_mu, var_mu = beta_mixture_moments(alphas, betas, pis)

        X_te = sample_from_mixture_beta(alphas, betas, pis, size=n_test)
        c_emp_te = np.array([np.mean(X_te), np.var(X_te)], dtype=float)

        S_eps = sample_random_S_eps(J=J_segments, domain_min=dom_min, domain_max=dom_max)
        Y_true = S_eps(S0_true_dependent(X_te, strategy, mean_mu, var_mu))

        Y_hat_ours = S_ours(X_te, c_emp_te)
        Y_hat_cf = S_cf(X_te)
        Y_hat_pgfr = pgfr_predictor(X_te, c_emp_te, n_pred=n_test)

        w2_ours.append(w2_1d_empirical(Y_hat_ours, Y_true))
        w2_cf.append(w2_1d_empirical(Y_hat_cf, Y_true))
        w2_pgfr.append(w2_1d_empirical(Y_hat_pgfr, Y_true))

    dW_ours = float(np.mean(w2_ours))
    dW_cf = float(np.mean(w2_cf))
    dW_pgfr = float(np.mean(w2_pgfr))

    return {
        "err_L2_ours": err_L2_ours,
        "err_L2_cf": err_L2_cf,
        "dW_ours": dW_ours,
        "dW_cf": dW_cf,
        "dW_pgfr": dW_pgfr,
        "pgfr_bandwidth": float(pgfr_info["bandwidth"]),
    }


# ============================================================
# Plot helpers
# ============================================================

def add_group_shading(ax, centers, halfspan, color="gray", alpha=0.06):
    for xc in centers:
        ax.axvspan(xc - halfspan, xc + halfspan, color=color, alpha=alpha, zorder=0)



def make_boxplot(ax, data_list, positions, width, color):
    return ax.boxplot(
        data_list,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor="none", edgecolor=color, linewidth=1.6),
        whiskerprops=dict(color=color, linewidth=1.4),
        capprops=dict(color=color, linewidth=1.4),
        medianprops=dict(color=color, linewidth=1.8),
        flierprops=dict(
            marker="o",
            markersize=4,
            markerfacecolor=color,
            markeredgecolor=color,
            alpha=0.55,
        ),
    )


# ============================================================
# Main two-panel plot
# ============================================================

def run_section_4_2_boxplot(
    n_obs=100,
    N_pairs=100,
    reps=10,
    pis=(0.3, 0.4, 0.3),
    strategies=("whitening", "global"),
    J_segments=8,
    m_bins=50,
    num_eval_mu=200,
    num_grid_x=800,
    n_test=1000,
    N_test_pairs=200,
    base_seed=2026,
    save_path="sec4_2_dependent_two_panel_boxplot.pdf",
    pgfr_bandwidth=None,
    pgfr_bandwidth_scale=1.0,
    pgfr_n_quantiles=200,
):
    res = {
        s: {
            "L2_ours": [],
            "L2_cf": [],
            "dW_ours": [],
            "dW_cf": [],
            "dW_pgfr": [],
        }
        for s in strategies
    }

    for s in strategies:
        for r in range(reps):
            seed = base_seed + 10000 * r + (0 if s == "whitening" else 777)
            out = run_one_replicate(
                strategy=s,
                n_obs=n_obs,
                N_pairs=N_pairs,
                pis=pis,
                J_segments=J_segments,
                m_bins=m_bins,
                num_eval_mu=num_eval_mu,
                num_grid_x=num_grid_x,
                n_test=n_test,
                N_test_pairs=N_test_pairs,
                seed=seed,
                pgfr_bandwidth=pgfr_bandwidth,
                pgfr_bandwidth_scale=pgfr_bandwidth_scale,
                pgfr_n_quantiles=pgfr_n_quantiles,
            )

            res[s]["L2_ours"].append(out["err_L2_ours"])
            res[s]["L2_cf"].append(out["err_L2_cf"])
            res[s]["dW_ours"].append(out["dW_ours"])
            res[s]["dW_cf"].append(out["dW_cf"])
            res[s]["dW_pgfr"].append(out["dW_pgfr"])

            print(
                f"[{s}] rep {r+1}/{reps} | "
                f"L2 ours={out['err_L2_ours']:.4g}, cf={out['err_L2_cf']:.4g} | "
                f"dW ours={out['dW_ours']:.4g}, cf={out['dW_cf']:.4g}, pgfr={out['dW_pgfr']:.4g} | "
                f"bw={out['pgfr_bandwidth']:.4g}"
            )

    # colors aligned with previous figures
    color_cf = "darkorange"
    color_ours = "blue"
    color_pgfr = "green"

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(strategies))
    labels = ["Whitening", "Global map"]

    # Left panel: map error
    ax = axes[0]
    width_left = 0.16
    offset_left = 0.14
    pos_ours_left = x - offset_left
    pos_cf_left = x + offset_left
    group_halfspan_left = offset_left + width_left / 2 + 0.08
    add_group_shading(ax, x, group_halfspan_left)

    make_boxplot(ax, [res[s]["L2_ours"] for s in strategies], pos_ours_left, width_left, color_ours)
    make_boxplot(ax, [res[s]["L2_cf"] for s in strategies], pos_cf_left, width_left, color_cf)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Vert\hat S_{n,N}-S_0\Vert_{L^2(\pi)}^2$")
    ax.grid(False)
    ax.set_xlim(-0.6, len(strategies) - 0.4)

    # Right panel: dW with PGFR
    ax = axes[1]
    width_right = 0.18
    offset_right = 0.22
    pos_ours_right = x - offset_right
    pos_cf_right = x
    pos_pgfr_right = x + offset_right
    group_halfspan_right = offset_right + width_right / 2 + 0.08
    add_group_shading(ax, x, group_halfspan_right)

    make_boxplot(ax, [res[s]["dW_ours"] for s in strategies], pos_ours_right, width_right, color_ours)
    make_boxplot(ax, [res[s]["dW_cf"] for s in strategies], pos_cf_right, width_right, color_cf)
    make_boxplot(ax, [res[s]["dW_pgfr"] for s in strategies], pos_pgfr_right, width_right, color_pgfr)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$d_{\mathcal{W}}(\hat{\nu},\nu)$")
    ax.grid(False)
    ax.set_xlim(-0.6, len(strategies) - 0.4)

    axes[0].set_title(r"Estimation error")
    axes[1].set_title(r"Prediction error")

    legend_handles = [
        Line2D([0], [0], color=color_ours, lw=2.2, label="Our method"),
        Line2D([0], [0], color=color_cf, lw=2.2, label="Ghodrati & Panaretos (2022)"),
        Line2D([0], [0], color=color_pgfr, lw=2.2, label="Tucker & Wu (2025)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=True,
        fontsize=18,
        handlelength=1.1,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, 1.04),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93], w_pad=2.0)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    return res


if __name__ == "__main__":
    run_section_4_2_boxplot(
        n_obs=100,
        N_pairs=100,
        reps=100,
        n_test=1000,
        N_test_pairs=1000,
        save_path="sec4_2_dependent_two_panel_boxplot.pdf",
        pgfr_bandwidth=None,
        pgfr_bandwidth_scale=1.0,
        pgfr_n_quantiles=200,
    )
