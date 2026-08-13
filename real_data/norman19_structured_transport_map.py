import numpy as np
from scipy.stats import gaussian_kde, wasserstein_distance
import matplotlib.pyplot as plt
import pickle

############################################################
# 1. 从 processed_data.pkl 构造分布回归的数据
############################################################

def build_data_from_results(results, n_samples=1000, seed=0):
    """
    从 processed_data.pkl 中构造：
      - X_ref: control KDE 采样 (reference distribution)
      - Y_all: 每个 perturbation KDE 采样 (N, n_samples)
      - C_all: 每个 perturbation 的 embedding 特征 (N, d)
      - pert_names
      - is_composite: 是否为复合扰动
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    perts = sorted(results.keys())
    single_perts, composite_perts = [], []

    for p in perts:
        if p == "control":
            continue
        rep = results[p]["representation"]
        if rep is None:
            continue
        if isinstance(rep, tuple):
            composite_perts.append(p)
        else:
            single_perts.append(p)

    # ==== control reference ====
    kde_control = results["control"]["kde"]
    X_ref = kde_control.resample(n_samples).reshape(-1)

    # ==== 采样每个 perturbation ====
    Y_all, C_all, pert_names, is_composite = [], [], [], []

    for p in single_perts + composite_perts:
        info = results[p]
        kde = info["kde"]
        rep = info["representation"]

        Y_i = kde.resample(n_samples).reshape(-1)

        if isinstance(rep, tuple):
            c_i = np.asarray(rep[0]) + np.asarray(rep[1])
            comp = True
        else:
            c_i = np.asarray(rep)
            comp = False

        pert_names.append(p)
        Y_all.append(Y_i)
        C_all.append(c_i)
        is_composite.append(comp)

    return (
        X_ref,
        np.array(Y_all),
        np.array(C_all),
        pert_names,
        np.array(is_composite, bool)
    )


############################################################
# 2. 划分训练 / 测试：测试 = only 复合扰动
############################################################

def split_train_test(is_composite, train_ratio_comp=0.8, seed=0):
    rng = np.random.default_rng(seed)
    comp_idx = np.where(is_composite)[0]
    single_idx = np.where(~is_composite)[0]

    rng.shuffle(comp_idx)

    n_train_comp = int(len(comp_idx) * train_ratio_comp)
    n_train_comp = max(1, min(n_train_comp, len(comp_idx) - 1))

    comp_train = comp_idx[:n_train_comp]
    comp_test  = comp_idx[n_train_comp:]

    train_ids = np.concatenate([single_idx, comp_train])
    test_ids  = comp_test

    return train_ids, test_ids


############################################################
# 3. build OT training pairs
############################################################

def build_train_data(X_ref, Y_all, C_all, train_ids, n_obs=None, seed=0):
    rng = np.random.default_rng(seed)
    if n_obs is None:
        n_obs = Y_all.shape[1]

    X_list, Y_list, C_list = [], [], []

    for idx in train_ids:
        ref_idx = rng.choice(len(X_ref), size=n_obs, replace=True)
        drug_idx = rng.choice(Y_all.shape[1], size=n_obs, replace=True)

        X_list.append(X_ref[ref_idx])
        Y_list.append(Y_all[idx, drug_idx])
        C_list.append(C_all[idx])

    return X_list, Y_list, C_list


############################################################
# 4. OT map helper
############################################################

def empirical_ot_map_1d(X, Y):
    xs = np.sort(X)
    ys = np.sort(Y)
    def Q(xq):
        return np.interp(xq, xs, ys, left=ys[0], right=ys[-1])
    return Q


############################################################
# 5. Weighted PAVA
############################################################

def weighted_pava(y, w, eps=1e-12):
    y = np.asarray(y)
    w = np.asarray(w) + eps
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
            sizes.append(s1 + s2)
            levels.append(lvl_new)
            weights.append(w_new)
    z = []
    for lvl, sz in zip(levels, sizes):
        z.extend([lvl] * sz)
    return np.array(z)


############################################################
# 6. 分布回归模型 S(x;c)
############################################################

def fit_global_S_with_c(X_list, Y_list, C_list, m_bins=50, max_iter=50):
    N = len(X_list)
    C = np.asarray(C_list)
    d = C.shape[1]

    # x-grid
    X_all = np.concatenate(X_list)
    xmin, xmax = X_all.min(), X_all.max()
    edges = np.linspace(xmin, xmax, m_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    w_ij = np.zeros((N, m_bins))
    y_ij = np.zeros((N, m_bins))

    for i in range(N):
        X, Y = X_list[i], Y_list[i]
        Q = empirical_ot_map_1d(X, Y)
        y_ij[i] = Q(centers)
        counts, _ = np.histogram(X, bins=edges)
        w_ij[i] = counts / len(X)

    eps = 1e-12

    # 初始化参数
    w0 = np.zeros(d)
    w1 = np.zeros(d)
    b0 = 1.0
    b1 = 0.0

    for it in range(max_iter):
        phi0 = C @ w0 + b0
        phi1 = C @ w1 + b1

        num = (w_ij * phi0[:, None] * (y_ij - phi1[:, None])).sum(axis=0)
        den = (w_ij * (phi0[:, None] ** 2)).sum(axis=0) + eps

        y_bar = np.zeros_like(den)
        ok = den > 0
        y_bar[ok] = num[ok] / den[ok]

        z = weighted_pava(y_bar, den)

        # 更新线性参数 (w0, b0, w1, b1)
        Z = z.reshape(1, -1)
        ZN = np.repeat(Z, N, axis=0)

        Y_flat = y_ij.flatten()
        W_flat = w_ij.flatten()
        C_tile = np.repeat(C, m_bins, axis=0)

        Z_expanded = ZN.reshape(-1)
        Phi1 = np.ones_like(Z_expanded)

        X_design = np.column_stack([
            C_tile * Z_expanded.reshape(-1, 1),  # for w0
            Z_expanded,                          # for b0
            C_tile,                              # for w1
            Phi1                                 # for b1
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
        x = np.asarray(x)
        base = np.interp(x, centers, z, left=z[0], right=z[-1])
        phi0 = c_vec @ w0 + b0
        phi1 = c_vec @ w1 + b1
        return phi0 * base + phi1

    return centers, z, S_hat


############################################################
# 7. 评估 W1
############################################################

def evaluate_split(split_ids, X_ref, Y_all, C_all, S_hat, n_eval=1000, seed=0):
    rng = np.random.default_rng(seed)
    W = []
    for idx in split_ids:
        ref_idx = rng.choice(len(X_ref), size=n_eval, replace=True)
        drug_idx = rng.choice(Y_all.shape[1], size=n_eval, replace=True)

        X_eval = X_ref[ref_idx]
        Y_true = Y_all[idx, drug_idx]
        Y_pred = S_hat(X_eval, C_all[idx])

        w = wasserstein_distance(Y_true, Y_pred)
        W.append(w)
    return np.array(W)


############################################################
# 8. 一键运行主入口
############################################################

def run_experiment_from_results(
        results,
        n_samples_kde=1000,
        train_ratio_comp=0.8,
        m_bins=50,
        max_iter=50,
        seed=0
):
    print("Building KDE data from processed_data.pkl ...")
    X_ref, Y_all, C_all, pert_names, is_composite = \
        build_data_from_results(results, n_samples=n_samples_kde, seed=seed)

    print("Splitting train/test (test only composite)...")
    train_ids, test_ids = split_train_test(is_composite, train_ratio_comp, seed)

    print("Building OT training pairs...")
    X_list, Y_list, C_list = build_train_data(
        X_ref, Y_all, C_all, train_ids,
        n_obs=n_samples_kde, seed=seed
    )

    print("Training distribution regression model ...")
    centers, z_hat, S_hat = fit_global_S_with_c(
        X_list, Y_list, C_list,
        m_bins=m_bins, max_iter=max_iter
    )

    print("Evaluating ...")
    train_w = evaluate_split(train_ids, X_ref, Y_all, C_all, S_hat,
                             n_eval=n_samples_kde, seed=seed+1)
    test_w = evaluate_split(test_ids, X_ref, Y_all, C_all, S_hat,
                            n_eval=n_samples_kde, seed=seed+2)

    print(f"Train W1: mean={train_w.mean():.4f}, median={np.median(train_w):.4f}")
    print(f"Test  W1: mean={test_w.mean():.4f}, median={np.median(test_w):.4f}")

    return {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "train_w": train_w,
        "test_w": test_w,
        "centers": centers,
        "z_hat": z_hat,
        "S_hat": S_hat,
        "pert_names": pert_names,
    }



from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

def plot_some_test_predictions(
    results,
    res,          # run_experiment_from_results 的返回值
    n_show=3,
    seed=0
):
    """
    在测试集上的 perturbation 上画预测结果：
      - ref:  results["control"]["kde"]
      - true: results[pert]["kde"]
      - pred: 用 S_hat 从 control 的原始样本 value 映射过去，再 KDE

    只从 test_ids 中选 perturbation 来画。
    """
    rng = np.random.default_rng(seed)

    pert_names = res["pert_names"]   # 与 Y_all / C_all 的顺序一致
    test_ids = res["test_ids"]       # 这些 index 对应于 test perturbations
    S_hat = res["S_hat"]

    # 从 test_ids 中拿到对应的 perturbation 名字
    test_perts = [pert_names[i] for i in test_ids]

    if len(test_perts) == 0:
        raise ValueError("test_ids 为空，没有测试集 perturbation 可以画")

    # 随机挑几个 test perturbation 来画（但在固定 seed 下是可复现的）
    rng.shuffle(test_perts)
    test_perts = test_perts[:min(n_show, len(test_perts))]

    # control 的 KDE 和原始样本（参考分布）
    ref_kde = results["control"]["kde"]
    ref_values = np.asarray(results["control"]["value"], dtype=float)

    # 固定 X_eval（保证每次运行结果一致）
    X_eval = np.sort(ref_values)

    xs = np.linspace(-10, 10, 2000)

    for pert_name in test_perts:
        info = results[pert_name]
        true_kde = info["kde"]
        rep = info["representation"]

        # 构造 c_vec：复合 perturbation 是 (vec_a, vec_b)
        if isinstance(rep, tuple):
            c_vec = np.asarray(rep[0]) + np.asarray(rep[1])
        else:
            c_vec = np.asarray(rep)

        # 预测样本（固定 X_eval -> 固定 Y_pred）
        Y_pred = S_hat(X_eval, c_vec)
        pred_kde = gaussian_kde(Y_pred)

        plt.figure(figsize=(5.5, 3.5))
        plt.plot(xs, ref_kde(xs),   label="Reference (control)",         lw=2, alpha=0.7)
        plt.plot(xs, true_kde(xs),  label="True test: " + pert_name,     lw=2)
        plt.plot(xs, pred_kde(xs),  label="Predicted on test",           lw=2, linestyle="--")
        plt.xlabel("PC1")
        plt.ylabel("Density")
        plt.title(f"[TEST] perturbation: {pert_name}")
        plt.legend()
        plt.tight_layout()
        plt.show()


############################################################
# 9. MAIN
############################################################

if __name__ == "__main__":
    with open("processed_data.pkl", "rb") as f:
        results = pickle.load(f)

    res = run_experiment_from_results(
        results,
        n_samples_kde=2000,
        train_ratio_comp=0.8,
        m_bins=50,
        max_iter=50,
        seed=2025,
    )

    plot_some_test_predictions(
        results=results,
        res=res,   # 前面 run_experiment_from_results 返回的
        n_show=3,
        seed=2025
    )