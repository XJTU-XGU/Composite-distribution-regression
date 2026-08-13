import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Basic building blocks
# ============================================================

def zeta_k(x, k, eta=1):
    """
    ζ_k(x):
    - ζ_0(x) = x
    - ζ_k(x) = x - sin(pi * k * x)/( |k| * pi ), for k != 0
    This is smooth, strictly increasing on [0,1], and maps 0->0, 1->1.
    x can be scalar or numpy array.
    """
    x = np.asarray(x)
    if k == 0:
        return x
    return x - eta * np.sin(np.pi * k * x) / (abs(k) * np.pi)


def S0_true_c(x, c, gamma=1.0):
    """
    Ground-truth regression operator with covariate c:
        S0(x; c) = c * ζ_4(x) + gamma * c,   c ~ U(-1,1)

    - x ∈ [0,1]  -> ζ_4(x) ∈ [0,1]
    - c ∈ [-1,1] -> S0(x;c) ∈ [-(1+gamma), 1+gamma]
    """
    x = np.asarray(x)
    c = np.asarray(c)
    return c * zeta_k(x, k=4) + gamma * c


def sample_random_S_eps(J=8,
                         K_choices=(-3, -2, -1, 1, 2, 3),
                         domain_min=0,
                         domain_max=2.0):
    """
    Construct a random monotone noise map S_eps: [domain_min, domain_max] -> [domain_min, domain_max].
    """
    # random breaks in (domain_min, domain_max)
    inner_breaks = np.sort(
        domain_min + np.random.rand(J - 1) * (domain_max - domain_min)
    )
    brks = np.concatenate(([domain_min], inner_breaks, [domain_max]))
    Ks = np.random.choice(K_choices, size=J)

    def S_eps(x):
        x = np.asarray(x)
        out = np.empty_like(x, dtype=float)
        for seg in range(J):
            a = brks[seg]
            b = brks[seg + 1]
            k = Ks[seg]
            if seg < J - 1:
                mask = (x >= a) & (x < b)
            else:
                mask = (x >= a) & (x <= b)
            if not np.any(mask):
                continue
            denom = (b - a) if (b - a) > 1e-12 else 1e-12
            s = (x[mask] - a) / denom
            y = zeta_k(s, k, eta=1.0)
            out[mask] = a + (b - a) * y
        return out

    return S_eps


# ============================================================
# 2. Sampling from mixture-of-3-Beta distributions
# ============================================================

def sample_mu_params():
    """
    Draw parameters for one μ_i:
    α_{i,j}, β_{i,j} ~ Uniform[1,10], for j=1,2,3.

    Returns:
    - alphas: shape (3,)
    - betas:  shape (3,)
    """
    alphas = np.random.uniform(1, 10, size=3)
    betas = np.random.uniform(1, 10, size=3)
    return alphas, betas


def sample_from_mixture_beta(alphas, betas, pis, size):
    """
    Sample 'size' points from the 3-component Beta mixture:
    f(x) = sum_{j=1}^3 π_j * Beta(α_j, β_j).

    Inputs:
    - alphas, betas: arrays shape (3,)
    - pis: mixture weights shape (3,), sum to 1
    - size: number of samples

    Output:
    - samples: shape (size,)
    """
    pis = np.asarray(pis)
    assert len(alphas) == 3 and len(betas) == 3 and len(pis) == 3

    comps = np.random.choice(3, size=size, p=pis)
    samples = np.zeros(size, dtype=float)
    for j in range(3):
        idx = (comps == j)
        n_j = np.sum(idx)
        if n_j > 0:
            samples[idx] = np.random.beta(alphas[j], betas[j], size=n_j)
    return samples


# ============================================================
# 3. Build per-pair data (μ_i, ν_i, c_i), but only via samples
#    Y = S_eps(S0_true_c(X, c_i))
# ============================================================

def generate_pair_samples(n_obs, pis, J_segments=8, gamma=0.5, domain_max=4.0):
    """
    Generate one training pair (μ_i, ν_i) with a covariate c_i, but we only
    keep n_obs samples from each for estimation.

    Steps:
    - Sample c_i ~ Uniform(-1,1).
    - Sample μ_i parameters (alphas, betas).
    - Sample X ~ μ_i, size n_obs (X ∈ [0,1]).
    - Sample a random noise map S_eps on [domain_min,domain_max],
      with domain_min = 0.
    - Compute Y = (S_eps ∘ S0(·;c_i))(X).

    Return:
    - X: shape (n_obs,), "observed samples from μ_i"
    - Y: shape (n_obs,), "observed samples from ν_i"
    - c_i: scalar covariate
    """
    # c_i ~ U(-1,1)
    c_i = 2.0 * np.random.rand()

    alphas, betas = sample_mu_params()
    X = sample_from_mixture_beta(alphas, betas, pis, size=n_obs)

    S_eps = sample_random_S_eps(J=J_segments,
                                domain_min=0,
                                domain_max=domain_max)
    Y = S_eps(S0_true_c(X, c_i, gamma=gamma))  # values in [domain_min, domain_max]

    return X, Y, c_i


# ============================================================
# 4. Empirical 1D OT map Q_i^n between μ_i and ν_i
# ============================================================

def empirical_ot_map_1d(X, Y):
    """
    Estimate the 1D optimal transport map Q_i^n from μ_i^n to ν_i^n.

    In 1D with equal mass samples, the OT map is the monotone match
    of quantiles:
        sort X -> xs, sort Y -> ys,
        map xs[k] -> ys[k].
    We then extend this mapping piecewise-linearly (monotone)
    and extrapolate constants at the boundaries.

    Inputs:
    - X: samples from μ_i (shape (n,))
    - Y: samples from ν_i (shape (n,)), with Y ~ pushforward of X under monotone map

    Outputs:
    - Q_fn(x_query): callable that returns interpolated ys value at x_query
    - xs_sorted: sorted X
    - ys_sorted: sorted Y
    """
    xs = np.sort(X)
    ys = np.sort(Y)

    def Q_fn(xq):
        return np.interp(xq, xs, ys, left=ys[0], right=ys[-1])

    return Q_fn, xs, ys


# ============================================================
# 5. Weighted PAVA (Pool-Adjacent-Violators Algorithm)
# ============================================================

def weighted_pava(y, w, eps=1e-12):
    """
    Solve the weighted isotonic regression problem:

        min_z Σ_j w_j * (y_j - z_j)^2
        s.t. z_1 <= z_2 <= ... <= z_m

    using PAVA with weights.

    Inputs:
    - y: array of length m (observed averages per bin)
    - w: array of length m (weights per bin, >=0)
    - eps: tiny number to avoid zero weight division by zero

    Output:
    - z_hat: array of length m, monotone nondecreasing
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float) + eps  # avoid zero division

    levels = []
    weights = []
    block_sizes = []

    for j in range(len(y)):
        levels.append(y[j])
        weights.append(w[j])
        block_sizes.append(1)

        # Merge backward while we violate monotonicity
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            lvl2 = levels.pop()
            lvl1 = levels.pop()
            w2 = weights.pop()
            w1 = weights.pop()
            sz2 = block_sizes.pop()
            sz1 = block_sizes.pop()

            new_w = w1 + w2
            new_lvl = (lvl1 * w1 + lvl2 * w2) / new_w
            new_sz = sz1 + sz2

            levels.append(new_lvl)
            weights.append(new_w)
            block_sizes.append(new_sz)

    z_hat = []
    for lvl, sz in zip(levels, block_sizes):
        z_hat.extend([lvl] * sz)

    return np.array(z_hat)


# ============================================================
# 6. Fit global regression map
#    S_hat(x;c) = φ0(c) z(x) + φ1(c)
#    with φ0(c) = a0*c + b0, φ1(c) = a1*c + b1
# ============================================================

def fit_global_S_with_c(X_list, Y_list, c_list, m_bins=50,
                        max_iter=50, tol=1e-5):
    """
    Core regression step with covariate c, for model

        S(x; c) = φ0(c) * z(x) + φ1(c),
        φ0(c) = a0 * c + b0,
        φ1(c) = a1 * c + b1.

    On grid x_j, we write:
        S(x_j; c_i) = (a0 * c_i + b0) * z_j + (a1 * c_i + b1).

    Objective:
        f(z,a0,b0,a1,b1) = Σ_i Σ_j w_ij [y_ij - φ0(c_i) z_j - φ1(c_i)]^2
        s.t. z_1 <= ... <= z_m

    Alternating minimization:
      1) Fix (a0,b0,a1,b1), solve for z by weighted PAVA.
      2) Fix z, solve for (a0,b0,a1,b1) by 4-dim weighted LS.
    """
    N = len(X_list)
    assert N == len(Y_list) == len(c_list)

    # grid in x
    edges = np.linspace(0.0, 1.0, m_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])  # length m_bins

    # accumulate weights and "responses" y_ij via empirical OT
    w_ij = np.zeros((N, m_bins), dtype=float)
    y_ij = np.zeros((N, m_bins), dtype=float)
    c_arr = np.asarray(c_list, dtype=float)  # can be in [-1,1]

    for i in range(N):
        X = X_list[i]
        Y = Y_list[i]

        Q_i, xs, ys = empirical_ot_map_1d(X, Y)
        y_ij[i, :] = Q_i(centers)

        counts, _ = np.histogram(X, bins=edges)
        w_ij[i, :] = counts.astype(float) / float(len(X))

    eps = 1e-12

    # ----- initialization: φ0(c)=1, φ1(c)=0 => 退化到原来的模型 -----
    a0, b0 = 0.0, 1.0   # φ0(c) = 1
    a1, b1 = 0.0, 0.0   # φ1(c) = 0

    phi0 = a0 * c_arr + b0   # shape (N,)
    phi1 = a1 * c_arr + b1   # shape (N,)

    phi0_mat = phi0[:, None]  # (N,1)
    phi1_mat = phi1[:, None]

    num0 = (w_ij * phi0_mat * (y_ij - phi1_mat)).sum(axis=0)
    den0 = (w_ij * (phi0_mat ** 2)).sum(axis=0) + eps

    y_bar0 = np.zeros_like(den0)
    mask0 = den0 > 0
    y_bar0[mask0] = num0[mask0] / den0[mask0]

    z = weighted_pava(y_bar0, den0, eps=eps)

    # ----- alternating updates -----
    for it in range(max_iter):
        z_old = z.copy()
        a0_old, b0_old, a1_old, b1_old = a0, b0, a1, b1

        # === Step 1: fix (a0,b0,a1,b1), update z via PAVA ===
        phi0 = a0 * c_arr + b0
        phi1 = a1 * c_arr + b1

        phi0_mat = phi0[:, None]   # (N,1)
        phi1_mat = phi1[:, None]   # (N,1)

        num = (w_ij * phi0_mat * (y_ij - phi1_mat)).sum(axis=0)
        den = (w_ij * (phi0_mat ** 2)).sum(axis=0) + eps

        y_bar = np.zeros_like(den)
        mask = den > 0
        y_bar[mask] = num[mask] / den[mask]

        z = weighted_pava(y_bar, den, eps=eps)

        # === Step 2: fix z, update (a0,b0,a1,b1) via weighted LS ===
        C = c_arr[:, None]          # (N,1)
        Z = z[None, :]              # (1,m_bins)

        # 让所有特征都变成 (N, m_bins)，再 ravel 成同一长度
        C_mat = np.repeat(C, m_bins, axis=1)   # (N,m_bins)
        Z_mat = np.repeat(Z, N, axis=0)        # (N,m_bins)

        F1 = (C_mat * Z_mat).ravel()           # c_i * z_j
        F2 = Z_mat.ravel()                     # z_j
        F3 = C_mat.ravel()                     # c_i
        F4 = np.ones_like(F1)                  # 1

        Y_flat = y_ij.ravel()
        W_flat = w_ij.ravel()

        mask_w = W_flat > 0
        if mask_w.sum() >= 4:
            sqrtW = np.sqrt(W_flat[mask_w])

            X_design = np.stack(
                [F1[mask_w], F2[mask_w], F3[mask_w], F4[mask_w]],
                axis=1
            )   # shape (M, 4)
            Xw = X_design * sqrtW[:, None]
            yw = Y_flat[mask_w] * sqrtW

            # solve least squares: beta = (a0,b0,a1,b1)
            use_ridge = (mask_w.sum() < 100)  # 或者根据条件数判断
            if use_ridge:
                lam = 1e0
                XtX = Xw.T @ Xw
                Xty = Xw.T @ yw
                beta = np.linalg.solve(XtX + lam * np.eye(4), Xty)
            else:
                beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            a0, b0, a1, b1 = beta
        else:
            # 权重信息太少，就保持参数不变
            pass

        # convergence check
        delta_z = np.max(np.abs(z - z_old))
        delta_params = max(abs(a0 - a0_old),
                           abs(b0 - b0_old),
                           abs(a1 - a1_old),
                           abs(b1 - b1_old))
        if max(delta_z, delta_params) < tol:
            break

    def S_hat_func(x, c):
        """
        Estimated regression operator S_hat(x;c) with
            S_hat(x;c) = (a0*c + b0) * z(x) + (a1*c + b1).
        """
        x = np.asarray(x)
        c = np.asarray(c)

        # base z(x) via interpolation
        base = np.interp(x.ravel(), centers, z,
                         left=z[0], right=z[-1]).reshape(x.shape)

        phi0_c = a0 * c + b0
        phi1_c = a1 * c + b1

        phi0_c = np.asarray(phi0_c)
        phi1_c = np.asarray(phi1_c)

        # broadcast to match base's shape
        while phi0_c.ndim < base.ndim:
            phi0_c = phi0_c[..., None]
            phi1_c = phi1_c[..., None]

        return phi0_c * base + phi1_c

    return centers, z, S_hat_func

# ============================================================
# 7. Compute L2 error between S_hat and S0 over x and c
#    || S_hat - S0 ||_{L^2([0,1]_x × [-1,1]_c)}
# ============================================================

def L2_error_over_xc(S_hat_func, num_grid_x=1000, num_grid_c=50, gamma=0.5):
    """
    Approximate
        || S_hat - S0 ||_{L^2([0,1]_x × [-1,1]_c)}
    = sqrt( (1/2) ∫_{-1}^1 ∫_0^1 |S_hat(x,c) - S0(x,c)|^2 dx dc )

    Numerically via trapezoid rule in x and c.
    """
    xs = np.linspace(0.0, 1.0, num_grid_x)
    cs = np.linspace(0.0, 2.0, num_grid_c)

    X, C = np.meshgrid(xs, cs, indexing="xy")  # shape (num_c, num_x)

    S0_vals = S0_true_c(X, C, gamma=gamma)
    Shat_vals = S_hat_func(X, C)
    diff2 = (Shat_vals - S0_vals) ** 2  # shape (num_c, num_x)

    # integrate over x first (axis=1, xs length)
    val_c = np.trapezoid(diff2, xs, axis=1)  # shape (num_c,)
    # then over c in [-1,1]
    val = np.trapezoid(val_c, cs)

    # divide by 2 = length of [-1,1], to get expectation over c
    return np.sqrt(val / 2.0)


# ============================================================
# 8. Single replicate of the experiment for given (n, N)
# ============================================================

def run_single_replicate(n_obs, N_pairs,
                         pis=(0.3, 0.4, 0.3),
                         J_segments=8,
                         m_bins=50,
                         gamma=0.5,
                         domain_max=1.5,
                         random_seed=None):
    """
    Run ONE Monte Carlo replicate with settings:
    - n_obs = n (sample size per distribution)
    - N_pairs = N (number of (μ_i, ν_i, c_i) pairs)

    Returns:
    - err_L2: float, L2 error between S_hat and S0 over x×c
    - centers, z_hat: the fitted base curve (for plotting)
    - S_hat_func: callable S_hat(x,c)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    X_list = []
    Y_list = []
    C_list = []
    for i in range(N_pairs):
        X_i, Y_i, c_i = generate_pair_samples(
            n_obs, pis,
            J_segments=J_segments,
            gamma=gamma,
            domain_max=domain_max
        )
        X_list.append(X_i)
        Y_list.append(Y_i)
        C_list.append(c_i)

    centers, z_hat, S_hat_func = fit_global_S_with_c(
        X_list, Y_list, C_list, m_bins=m_bins
    )

    err = L2_error_over_xc(S_hat_func, gamma=gamma)

    return err, centers, z_hat, S_hat_func


# ============================================================
# 9. Run full grid of (n, N) with multiple reps + boxplots
#    And when showing S0 vs S_hat, show for c=-0.5, 0, 0.5
# ============================================================

def run_full_experiment(
    n_list=(10, 100, 1000),
    N_list=(10, 100, 1000),
    reps=50,
    pis=(0.3, 0.4, 0.3),
    J_segments=8,
    m_bins=50,
    gamma=0.5,
    domain_max=1.5,
    base_seed=1234,
    make_plots=True
):
    """
    For each combination (n, N), run 'reps' independent replicates.
    Store the L2 errors and create boxplots.

    When plotting S0 vs S_hat, we show three c-slices:
        c = -0.5, 0.0, 0.5.
    """
    results = {}

    for n_obs in n_list:
        for N_pairs in N_list:

            errs = []
            last_centers, last_z_hat, last_S_hat_func = None, None, None

            for r in range(reps):
                seed = base_seed + 100000 * n_obs + 1000 * N_pairs + r
                err, centers, z_hat, S_hat_func = run_single_replicate(
                    n_obs=n_obs,
                    N_pairs=N_pairs,
                    pis=pis,
                    J_segments=J_segments,
                    m_bins=m_bins,
                    gamma=gamma,
                    domain_max=domain_max,
                    random_seed=seed
                )
                errs.append(err)
                last_centers, last_z_hat, last_S_hat_func = centers, z_hat, S_hat_func

            results[(n_obs, N_pairs)] = errs
            print(f"(n={n_obs}, N={N_pairs})  mean L2 err = {np.mean(errs):.4f}  "
                  f"median={np.median(errs):.4f}")

            # Plot one representative fit: show S0 and S_hat for c=-0.5,0,0.5
            if make_plots and last_S_hat_func is not None:
                xs_plot = np.linspace(0, 1, 400)
                c_vals = [0.5, 1.0, 1.5]

                fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
                for idx, c_val in enumerate(c_vals):
                    ax = axes[idx]
                    S0_curve = S0_true_c(xs_plot, c_val, gamma=gamma)
                    Shat_curve = last_S_hat_func(xs_plot, c_val)

                    ax.plot(xs_plot, S0_curve,
                            label="True S0(x;c)", lw=2)
                    ax.plot(xs_plot, Shat_curve,
                            label=r"Fit $\hat S(x;c)$", lw=2,
                            linestyle="--")
                    ax.set_title(f"c = {c_val:.2f}")
                    ax.set_xlabel("x")
                    ax.grid(True)
                    if idx == 0:
                        ax.set_ylabel("S(x;c)")
                        ax.legend()

                plt.suptitle(f"One replicate (n={n_obs}, N={N_pairs})")
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()

    # Boxplots: for each fixed n, show errors vs N
    if make_plots:
        for n_obs in n_list:
            fig, ax = plt.subplots(figsize=(4, 3))
            data_to_plot = []
            xticks = []
            for N_pairs in N_list:
                data_to_plot.append(results[(n_obs, N_pairs)])
                xticks.append(f"N={N_pairs}")
            ax.boxplot(data_to_plot, labels=xticks)
            ax.set_title(f"L2 Error vs N  (n={n_obs})")
            ax.set_ylabel("||S_hat - S0||_L2 over (x,c)")
            plt.tight_layout()
            plt.show()

    return results


# ============================================================
# 10. Example main run
# ============================================================

if __name__ == "__main__":
    """
    You can run this file directly.
    It will:
    - run a smaller experiment (reps=20 just for speed),
    - print mean/median L2 errors,
    - show sanity-check plots for c = -0.5, 0, 0.5.
    """
    results = run_full_experiment(
        n_list=(10, 100, 10),
        N_list=(10, 100, 10),
        reps=20,          # use 100 for more stable boxplots, but it's slower
        pis=(0.3, 0.4, 0.3),
        J_segments=8,
        m_bins=50,
        gamma=0.5,
        domain_max=3.0,   # should be >= 1+gamma
        base_seed=2025,
        make_plots=True,
    )
