from Structured_transport_map_core import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"          # 让公式更像 Times 系
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18,              # base font size
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 18,
    "figure.titlesize": 18,
    "mathtext.fontset": "stix",   # math like Times
    "axes.unicode_minus": False,
})

def kde_1d_gaussian(samples, grid, bw=None, eps=1e-12):
    """
    纯 numpy 的 1D 高斯核 KDE（不依赖 scipy）。

    samples: (n,)
    grid: (G,)
    bw: bandwidth；None 时用 Silverman rule
    """
    x = np.asarray(samples, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.zeros_like(grid, dtype=float)

    std = np.std(x) + eps
    if bw is None:
        bw = 1.06 * std * (n ** (-1/5)) + eps  # Silverman

    u = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * u * u).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
    return dens


def visualize_3x3_with_multi_eps(
    n_obs=1200,
    pis=(0.3, 0.4, 0.3),
    c_vals=(0.5, 1.0, 1.5),
    J_segments=8,
    gamma=0.5,
    domain_max=3.0,
    seed=2018,
    grid_size=400,
    scale_to_unit=True,
    response_reps=6,          # 每个格子画几条 response（不同 epsilon）
    show_response_mean=True,  # 是否加粗画均值曲线
    response_alpha=1.0,      # 单条 response 的透明度
    ylim_quantile=0.99,       # 用全图密度点的分位数设 y 上限
    ylim_pad=1.10             # 上限再乘一点 padding
):
    np.random.seed(seed)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharex=True, sharey=True)

    # x轴网格（参考图是[0,1]）
    if scale_to_unit:
        grid = np.linspace(0.0, 1.0, grid_size)
    else:
        grid = np.linspace(0.0, domain_max, grid_size)

    # 用来统一设 y 轴范围
    all_density_vals = []
    mu_style = dict(color="tab:blue", linestyle="-", lw=1.6)
    nu_style = dict(color="tab:orange", linestyle="--", lw=1.6)

    for i in range(1):
        # ---- 固定该行的 μ_i：只抽一次参数 + 只采一次 X ----
        alphas_i, betas_i = sample_mu_params()
        X_row = sample_from_mixture_beta(alphas_i, betas_i, pis, size=n_obs)

        # predictor 密度（蓝）
        if scale_to_unit:
            X_plot = X_row
        else:
            X_plot = X_row * domain_max

        dens_X = kde_1d_gaussian(X_plot, grid)
        all_density_vals.append(dens_X)

        for j, c in enumerate(c_vals):
            ax = axes[j]

            # 先画 predictor（蓝）
            ax.plot(grid, dens_X, label=r"$\mu$" if (i == 0 and j == 0) else None, **mu_style)

            # 多条 response（橙，透明）
            dens_Y_list = []
            for r in range(response_reps):
                S_eps = sample_random_S_eps(J=J_segments, domain_min=0.0, domain_max=domain_max)
                Y = S_eps(S0_true_c(X_row, c, gamma=gamma))

                if scale_to_unit:
                    Y_plot = Y / domain_max
                else:
                    Y_plot = Y

                dens_Y = kde_1d_gaussian(Y_plot, grid)
                dens_Y_list.append(dens_Y)
                all_density_vals.append(dens_Y)

                ax.plot(
                    grid,
                    dens_Y,
                    alpha=response_alpha,
                    label=r"$\nu$" if (i == 0 and j == 0 and r == 0) else None,
                    **nu_style,
                )

            # 画 response 的均值曲线（更像“参考图”的单条橙线，但保留多条样本不确定性）
            if show_response_mean and len(dens_Y_list) > 0:
                dens_mean = np.mean(np.stack(dens_Y_list, axis=0), axis=0)
                ax.plot(grid, dens_mean, color="tab:red", linestyle="-.", lw=2.5)

            ax.grid(True, alpha=0.25)
            ax.set_xlim(grid[0], grid[-1])

            if i==0 and j==0:
                ax.legend(fontsize=20)

            if i == 0:
                ax.set_title(f"$z = {c:g}$", fontsize=20)
            if j == 0:
                ax.set_ylabel(f"Density", fontsize=20)
            ax.set_xlabel(f"$x$", fontsize=20)

    # ---- 统一压缩纵轴范围：用全图密度点的分位数当上限 ----
    dens_all = np.concatenate([d.ravel() for d in all_density_vals])
    y_max = np.quantile(dens_all, ylim_quantile) * ylim_pad
    if not np.isfinite(y_max) or y_max <= 0:
        y_max = np.max(dens_all) if np.max(dens_all) > 0 else 1.0

    for ax in axes.ravel():
        ax.set_ylim(0.0, y_max)

    supt = (r"3×3 densities: predictor $\mu_i$ (blue solid) vs responses "
            r"$\nu_{i,z}^{(\ell)}=S_{\varepsilon}^{(\ell)}\#(S_0(\cdot,z)\#\mu_i)$ (orange dashed)")
    if scale_to_unit:
        supt += "\n(response scaled by domain_max for plotting on [0,1])"
    supt += f"\n(each panel: {response_reps} eps samples; y-limit={int(ylim_quantile*100)}% quantile)"

    # fig.suptitle(supt, y=1.02, fontsize=13)
    plt.tight_layout()
    os.makedirs("./outputs", exist_ok=True)
    plt.savefig(os.path.join("./outputs", "Figure1.pdf"))
    plt.show()


# 示例运行
if __name__ == "__main__":
    visualize_3x3_with_multi_eps(
        n_obs=1200,
        pis=(0.3, 0.4, 0.3),
        c_vals=(0.8, 1.0, 1.2),
        J_segments=8,
        gamma=0.5,
        domain_max=3.0,
        seed=2026,
        response_reps=1,
        ylim_quantile=0.9999,
        show_response_mean=False,
    )
