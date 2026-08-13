from Structured_transport_map_core import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import os

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18,              # base font size
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    # "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 18,
    "figure.titlesize": 18,
    "mathtext.fontset": "stix",   # math like Times
    "axes.unicode_minus": False,
})


def fit_one_replicate_only(
    n_obs, N_pairs,
    pis=(0.3, 0.4, 0.3),
    J_segments=8,
    m_bins=50,
    gamma=0.5,
    domain_max=3.0,
    random_seed=None
):
    if random_seed is not None:
        np.random.seed(random_seed)

    X_list, Y_list, C_list = [], [], []
    for _ in range(N_pairs):
        X_i, Y_i, c_i = generate_pair_samples(
            n_obs=n_obs,
            pis=pis,
            J_segments=J_segments,
            gamma=gamma,
            domain_max=domain_max
        )
        X_list.append(X_i)
        Y_list.append(Y_i)
        C_list.append(c_i)

    _, _, S_hat_func = fit_global_S_with_c(
        X_list, Y_list, C_list, m_bins=m_bins
    )
    return S_hat_func


def plot_3x3_true_black_hat_colored_spaghetti(
    n_list=(20, 200),
    N_list=(20, 200),
    c_vals=(0.8, 1.0, 1.2),
    reps=10,                         # 每种设置画 10 条（估计曲线）
    pis=(0.3, 0.4, 0.3),
    J_segments=8,
    m_bins=50,
    gamma=0.5,
    domain_max=3.0,
    base_seed=2025,
    xs_plot=None,
    scale_to_unit=False,
    lw_true=2.0,
    lw_hat=0.3,
    alpha_hat=0.6,
    show_legend=True,
    save_path="Figure2.pdf"
):
    """
    Grid plot (row=n, col=N):
      - True S0: black, linewidth=2.0, styles: '-', '--', '-.' for different c
      - Hat S: different colors and line styles for different c
      - Only curves, no mean/band; reps=10 spaghetti per (n,N)
    """
    if xs_plot is None:
        xs_plot = np.linspace(0.0, 1.0, 400)

    assert len(c_vals) == 3, "此版本默认 3 个 c"
    colors = ["tab:blue", "tab:orange", "tab:green"]
    linestyles = ["-", "--", "-."]
    markers = ["^", "o", "*"]  # 对应 '-', '-o', '-*'

    fig, axes = plt.subplots(len(n_list), len(N_list),
                             figsize=(10.0, 7.2),
                             sharex=True, sharey=True)

    for i, n_obs in enumerate(n_list):
        for j, N_pairs in enumerate(N_list):
            ax = axes[i, j]

            # --- 再画估计（每个 (n,N) 重复 reps 次；每次对3个c都画） ---
            for r in range(reps):
                seed = base_seed + 100000 * n_obs + 1000 * N_pairs + r
                S_hat = fit_one_replicate_only(
                    n_obs=n_obs, N_pairs=N_pairs,
                    pis=pis, J_segments=J_segments, m_bins=m_bins,
                    gamma=gamma, domain_max=domain_max,
                    random_seed=seed
                )

                for c, col, ls in zip(c_vals, colors, linestyles):
                    y_hat = S_hat(xs_plot, c)
                    if scale_to_unit:
                        y_hat = y_hat / domain_max

                    ax.plot(xs_plot, y_hat,
                            color=col, lw=lw_hat, linestyle=ls,
                            alpha=alpha_hat)

            ax.set_title(fr"$n={n_obs},\,N={N_pairs}$", fontsize=18)
            ax.set_ylim([0.3,2.0])
            ax.grid(True, alpha=0.25)

            if i == len(n_list) - 1:
                ax.set_xlabel("$x$", fontsize=18)

            # if j == 0:
            #     ax.set_ylabel(
            #         fr"$S(x,c)/{domain_max}$" if scale_to_unit else r"$S(x,c)$",
            #         fontsize=18
            #     )

            # --- 先画真值（每个 c 一条） ---
            for c, mk, ls in zip(c_vals, markers, linestyles):
                y_true = S0_true_c(xs_plot, c, gamma=gamma)
                if scale_to_unit:
                    y_true = y_true / domain_max

                ax.plot(
                    xs_plot, y_true,
                    color="black", lw=lw_true, linestyle=ls,
                    marker=mk,
                    markevery=40,  # 控制 marker 密度（避免太密）
                    markersize=5
                )

    # --- 全局 legend（6项）：按 c 成对排列，真值在前、估计在后 ---
    if show_legend:
        handles = []
        labels = []

        for c, mk, col, ls in zip(c_vals, markers, colors, linestyles):
            handles.append(Line2D([0], [0], color="black", lw=lw_true,
                                  linestyle=ls, marker=mk, markersize=6))
            labels.append(fr"$S_0(\cdot,{c:g})$")
            handles.append(Line2D([0], [0], color=col, lw=max(lw_hat, 1.4), linestyle=ls))
            labels.append(fr"$\hat S_{{n,N}}(\cdot,{c:g})$")

        fig.legend(handles, labels,
                   loc="upper center",
                   bbox_to_anchor=(0.5, 1.01),
                   ncol=3,
                   frameon=True,
                   fontsize=18,
                   handlelength=1.8,
                   handletextpad=0.3,
                   columnspacing=0.9,
                   labelspacing=0.35)

    plt.tight_layout(rect=[0.01, 0.01, 0.98, 0.88])
    os.makedirs("./outputs",exist_ok=True)
    plt.savefig(os.path.join("./outputs",save_path))
    plt.show()


if __name__ == "__main__":
    plot_3x3_true_black_hat_colored_spaghetti(
        n_list=(10, 100),
        N_list=(10, 100),
        c_vals=(0.8, 1.0, 1.2),
        reps=100,                 # 每种设置 10 条
        pis=(0.3, 0.4, 0.3),
        J_segments=8,
        m_bins=50,
        gamma=0.5,
        domain_max=3.0,
        base_seed=2026,
        scale_to_unit=False,
        save_path="Figure2.pdf"
    )
