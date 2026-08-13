import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from leave_one_out import *
from norman19_structured_transport_map import *

import matplotlib as mpl
import matplotlib.ticker as mticker


# --- Global font settings (Times New Roman + size 15) ---
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

CURVE_STYLES = {
    "response": dict(color="black", lw=1.5, ls="-", alpha=0.95),
    "ours": dict(color="blue", lw=1.5, ls="--", alpha=0.95),
    "otm": dict(color="darkorange", lw=1.5, ls="-.", alpha=0.95),
    "diag": dict(color="gray", lw=1.0, ls=":", alpha=0.95),
}


def _safe_kde(samples, seed=0, jitter=1e-6, bw_method=None):
    x = np.asarray(samples, float).ravel()
    if x.size < 2:
        x = np.concatenate([x, x + 1e-3])
    if np.std(x) < 1e-10:
        rng = np.random.default_rng(seed)
        x = x + rng.normal(scale=jitter, size=x.shape)
    return gaussian_kde(x, bw_method=bw_method)


def plot_folds_grid_density_only(
    results,
    fold_names,
    n_samples_kde=2000,
    n_obs=None,
    m_bins=50,
    max_iter=50,
    seed=2026,
    x_min=-10,
    x_max=10,
    n_grid=1200,
    save_path="fold_plots_density_only/fold_grid_density_only.png",
    save=True,
    show=True,
):
    import os

    if n_obs is None:
        n_obs = n_samples_kde

    X_ref, Y_all, C_all, pert_names, is_composite = build_data_from_results(
        results, n_samples=n_samples_kde, seed=seed
    )
    name2idx = {p: i for i, p in enumerate(pert_names)}
    all_ids = np.arange(len(pert_names))

    xs = np.linspace(x_min, x_max, n_grid)
    ref_values = np.asarray(results["control"]["value"], dtype=float)
    X_eval = np.sort(ref_values)

    style = CURVE_STYLES

    cached = []
    top_max = 0.0

    for fold_name in fold_names:
        if fold_name not in results:
            raise KeyError(f"results 中没有 {fold_name}")
        if fold_name not in name2idx:
            raise KeyError(f"{fold_name} 未进入 pert_names（可能 representation=None 被过滤）")

        test_id = name2idx[fold_name]
        train_ids = all_ids[all_ids != test_id]

        X_list, Y_list, C_list = build_train_data(
            X_ref, Y_all, C_all, train_ids,
            n_obs=n_obs, seed=seed + 100 + test_id
        )

        _, _, S_hat = fit_global_S_with_c(
            X_list, Y_list, C_list, m_bins=m_bins, max_iter=max_iter
        )
        _, _, T_hat = fit_global_T_no_c(X_list, Y_list, m_bins=m_bins)

        true_kde = results[fold_name]["kde"]

        rep = results[fold_name]["representation"]
        if isinstance(rep, tuple):
            c_vec = np.asarray(rep[0]) + np.asarray(rep[1])
        else:
            c_vec = np.asarray(rep)

        Y_pred_ours = S_hat(X_eval, c_vec)
        kde_ours = _safe_kde(Y_pred_ours, seed=seed + 1000 + test_id)

        Y_pred_T = T_hat(X_eval)
        kde_T = _safe_kde(Y_pred_T, seed=seed + 2000 + test_id)

        top_vals = np.vstack([true_kde(xs), kde_ours(xs), kde_T(xs)])
        top_max = max(top_max, float(np.max(top_vals)))

        cached.append((fold_name, true_kde, kde_ours, kde_T))

    x_ticks = np.linspace(x_min, x_max, 5)
    top_ylim = (0.0, top_max * 1.05 + 1e-12)
    top_yticks = np.linspace(top_ylim[0], top_ylim[1], 5)

    K = len(fold_names)
    fig, axes = plt.subplots(1, K, figsize=(4 * K, 3.6), sharex=False, sharey=False)

    if K == 1:
        axes = np.array([axes])

    legend_handles = None
    legend_labels = None

    for j, (fold_name, true_kde, kde_ours, kde_T) in enumerate(cached):
        ax = axes[j]

        h1, = ax.plot(xs, true_kde(xs), label="Response", **style["response"])
        h2, = ax.plot(xs, kde_ours(xs), label="Prediction (CSOT)", **style["ours"])
        h3, = ax.plot(xs, kde_T(xs), label="Prediction (OTM)", **style["otm"])

        ax.set_title(fold_name)
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

        ax.set_ylim(*top_ylim)
        ax.set_yticks(top_yticks)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.grid(True, alpha=0.25)

        if j == 0:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)

        if legend_handles is None:
            legend_handles = [h1, h2, h3]
            legend_labels = ["Response", "Prediction (Our method)", "Prediction (Ghodrati & Panaretos (2022))"]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        borderaxespad=0.0,
        handlelength=2.6,
        columnspacing=1.6,
    )

    fig.supxlabel("x", y=0.03)
    fig.subplots_adjust(top=0.78, bottom=0.20, wspace=0.25)

    if save:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Saved] {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------
# Example main
# -----------------------------
if __name__ == "__main__":
    import pickle

    with open("processed_data.pkl", "rb") as f:
        results = pickle.load(f)

    folds = ["PLK4_STIL", "KIF18B_KIF2C", "BCL2L11_BAK1"]

    plot_folds_grid_density_only(
        results=results,
        fold_names=folds,
        n_samples_kde=2000,
        n_obs=2000,
        m_bins=50,
        max_iter=50,
        seed=2026,
        x_min=-5,
        x_max=10,
        n_grid=2000,
        save_path="outputs/Figure7.pdf",
        save=True,
        show=True,
    )
