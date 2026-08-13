import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union


# ============================================================
# Basic utilities
# ============================================================

def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def quantile_grid(n_quantiles: int = 200, eps: float = 1e-2) -> np.ndarray:
    """Uniform grid in (0,1) used to represent 1D quantile functions."""
    if not (0.0 < eps < 0.5):
        raise ValueError("eps must lie in (0, 0.5).")
    return np.linspace(eps, 1.0 - eps, int(n_quantiles))


def samples_to_quantiles(sample_list: Sequence[np.ndarray], u_grid: np.ndarray) -> np.ndarray:
    """Convert a list of 1D empirical samples into quantile-function rows."""
    u_grid = np.asarray(u_grid, dtype=float)
    return np.vstack([
        np.quantile(np.asarray(s, dtype=float), u_grid) for s in sample_list
    ])


def quantile_to_sorted_samples(q: np.ndarray, u_grid: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Convert a quantile curve into a sorted pseudo-sample of size n_samples.
    This is deterministic and convenient for empirical 1D W2 evaluation.
    """
    u_eval = np.linspace(u_grid[0], u_grid[-1], int(n_samples))
    return np.interp(u_eval, u_grid, np.asarray(q, dtype=float))


def w2_quantile_distance(q1: np.ndarray, q2: np.ndarray, u_grid: np.ndarray) -> float:
    """1D W2 distance between two distributions represented by quantiles."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    u_grid = np.asarray(u_grid, dtype=float)
    return float(np.sqrt(np.trapezoid((q1 - q2) ** 2, x=u_grid)))


def _pava_increasing(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted projection onto the cone of nondecreasing sequences via PAVA."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    vals = y.copy()
    wgts = w.copy()
    starts = np.arange(n)
    ends = np.arange(n)
    m = n

    i = 0
    while i < m - 1:
        if vals[i] <= vals[i + 1] + 1e-15:
            i += 1
            continue

        new_w = wgts[i] + wgts[i + 1]
        new_v = (wgts[i] * vals[i] + wgts[i + 1] * vals[i + 1]) / max(new_w, 1e-15)

        vals[i] = new_v
        wgts[i] = new_w
        ends[i] = ends[i + 1]

        vals[i + 1:m - 1] = vals[i + 2:m]
        wgts[i + 1:m - 1] = wgts[i + 2:m]
        starts[i + 1:m - 1] = starts[i + 2:m]
        ends[i + 1:m - 1] = ends[i + 2:m]
        m -= 1

        while i > 0 and vals[i - 1] > vals[i] + 1e-15:
            new_w = wgts[i - 1] + wgts[i]
            new_v = (wgts[i - 1] * vals[i - 1] + wgts[i] * vals[i]) / max(new_w, 1e-15)
            vals[i - 1] = new_v
            wgts[i - 1] = new_w
            ends[i - 1] = ends[i]

            vals[i:m - 1] = vals[i + 1:m]
            wgts[i:m - 1] = wgts[i + 1:m]
            starts[i:m - 1] = starts[i + 1:m]
            ends[i:m - 1] = ends[i + 1:m]
            m -= 1
            i -= 1

    out = np.empty(n, dtype=float)
    for j in range(m):
        out[starts[j]:ends[j] + 1] = vals[j]
    return out


def median_heuristic_bandwidth(Zq: np.ndarray, u_grid: np.ndarray) -> float:
    """Median of pairwise W2 distances between predictor distributions."""
    n = Zq.shape[0]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = w2_quantile_distance(Zq[i], Zq[j], u_grid)
            if d > 0:
                dists.append(d)
    if len(dists) == 0:
        return 1.0
    return float(np.median(dists))


def _coerce_covariates(c_list: Union[np.ndarray, Sequence[Any]]) -> np.ndarray:
    """
    Convert scalar/vector covariates into an array of shape (n, p).
    Supports:
      - list of scalars
      - 1D numpy array of length n
      - list of 1D arrays / list of lists
      - 2D numpy array of shape (n, p)
    """
    if isinstance(c_list, np.ndarray):
        arr = np.asarray(c_list, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        if arr.ndim == 2:
            return arr
        raise ValueError("Covariates must be 1D or 2D.")

    if len(c_list) == 0:
        raise ValueError("c_list must be non-empty.")

    first = np.asarray(c_list[0], dtype=float)
    if first.ndim == 0:
        return np.asarray(c_list, dtype=float).reshape(-1, 1)

    rows = [np.asarray(c, dtype=float).reshape(-1) for c in c_list]
    p = rows[0].shape[0]
    for row in rows:
        if row.shape[0] != p:
            raise ValueError("All covariate vectors must have the same dimension.")
    return np.vstack(rows)


# ============================================================
# Partially-global Frechet regression for 1D distribution response
# ============================================================

@dataclass
class PartiallyGlobalFrechet1D:
    """
    Local-constant partially-global Frechet regression specialized to
      - Euclidean predictor X in R^p
      - metric predictor Z
      - 1D distribution-valued response Y represented by quantile functions

    This is a 1D-Wasserstein implementation of the estimator in Tucker & Wu (2021),
    specialized to the mixed-predictor local-constant setting.

    Important implementation choice:
    - We store local smoother rows normalized to sum to 1.
    - With that convention, the paper's empirical criterion induces the signed weights
          a(x,z) = s(z) + (alpha - S^T alpha) / n,
      where alpha_i = C_i^T w2^{-1} w1(x,z).
    """
    bandwidth: float
    delta: Callable[[Any, Any], float]
    kernel: Callable[[np.ndarray], np.ndarray] = gaussian_kernel
    ridge: float = 1e-8

    def _kernel_weights_from_distances(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        k = self.kernel(d / max(self.bandwidth, 1e-12))
        s = np.sum(k)
        if (not np.isfinite(s)) or s <= 0:
            out = np.zeros_like(d)
            out[np.argmin(d)] = 1.0
            return out
        return k / s

    def _local_weights(self, z: Any) -> np.ndarray:
        d = np.array([self.delta(zi, z) for zi in self.Z], dtype=float)
        return self._kernel_weights_from_distances(d)

    def fit(
        self,
        X: np.ndarray,
        Z: Sequence[Any],
        Y_quantiles: np.ndarray,
        u_grid: np.ndarray,
    ):
        X = np.asarray(X, dtype=float)
        Y_quantiles = np.asarray(Y_quantiles, dtype=float)
        u_grid = np.asarray(u_grid, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must have shape (n, p).")
        if Y_quantiles.ndim != 2:
            raise ValueError("Y_quantiles must have shape (n, m).")

        n, p = X.shape
        if len(Z) != n or Y_quantiles.shape[0] != n:
            raise ValueError("X, Z, and Y_quantiles must have matching first dimension.")
        if Y_quantiles.shape[1] != u_grid.shape[0]:
            raise ValueError("Y_quantiles.shape[1] must equal len(u_grid).")

        self.X = X
        self.Z = list(Z)
        self.Yq = Y_quantiles
        self.u_grid = u_grid
        self.n = n
        self.p = p

        # S_train[i, j] = normalized local-constant weight of sample j at query Z_i.
        self.S_train = np.vstack([self._local_weights(z_i) for z_i in self.Z])

        # 8[X | Z_i] using normalized smoother rows.
        self.EX_given_Z = self.S_train @ self.X

        # C_i = X_i - 8[X|Z_i]
        self.centered_X = self.X - self.EX_given_Z

        # 72 = n^{-1} sum_i C_i C_i^T
        w2 = (self.centered_X.T @ self.centered_X) / float(n)
        w2 = w2 + self.ridge * np.eye(p)
        self.inv_w2 = np.linalg.pinv(w2)
        return self

    def _signed_weights(self, x: np.ndarray, z: Any) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.p:
            raise ValueError(f"x must have length {self.p}.")

        # normalized local smoother at the target z
        s = self._local_weights(z)

        # 71(x,z) = x - sum_i s_i(z) X_i
        w1 = x - s @ self.X

        # alpha_i = C_i^T 72^{-1} 71(x,z)
        alpha = self.centered_X @ self.inv_w2 @ w1

        # Exact induced signed weights after expanding the criterion with normalized S.
        a = s + (alpha - self.S_train.T @ alpha) / float(self.n)

        ssum = np.sum(a)
        if ssum <= 1e-12:
            # Fallback to the local component; in theory sum(a)=1 under exact arithmetic.
            a = s.copy()
            ssum = np.sum(a)
        a = a / ssum
        return a

    def predict_quantile(self, x: np.ndarray, z: Any) -> np.ndarray:
        a = self._signed_weights(x, z)
        q_bar = a @ self.Yq
        q_hat = _pava_increasing(q_bar)
        return q_hat

    def predict_samples(self, x: np.ndarray, z: Any, n_samples: int) -> np.ndarray:
        q_hat = self.predict_quantile(x, z)
        return quantile_to_sorted_samples(q_hat, self.u_grid, n_samples=n_samples)

    def predict(self, X_new: np.ndarray, Z_new: Sequence[Any]) -> np.ndarray:
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new[None, :]
        if X_new.shape[0] != len(Z_new):
            raise ValueError("X_new and Z_new must have the same number of samples.")
        return np.vstack([
            self.predict_quantile(x, z) for x, z in zip(X_new, Z_new)
        ])


# ============================================================
# Convenience wrapper for your simulation setting
# ============================================================

def fit_pgfr_baseline_from_samples(
    X_list: Sequence[np.ndarray],
    Y_list: Sequence[np.ndarray],
    c_list: Union[np.ndarray, Sequence[Any]],
    bandwidth: Optional[float] = None,
    bandwidth_scale: float = 1.0,
    n_quantiles: int = 200,
    quantile_eps: float = 1e-2,
    ridge: float = 1e-8,
):
    """
    Convenience wrapper for the conditional distribution-to-distribution baseline:
        Euclidean predictor  : c  (scalar or vector)
        Metric predictor     : input distribution X
        Response distribution: Y

    Returns
    -------
    predictor : callable
        predictor(X_new_samples, c_new, n_pred=None) -> sorted pseudo-sample of predicted Y
    info : dict
        Stores the fitted model and representations for reuse.
    """
    if len(X_list) == 0:
        raise ValueError("X_list must be non-empty.")
    if not (len(X_list) == len(Y_list) == len(c_list)):
        raise ValueError("X_list, Y_list, c_list must have the same length.")

    u_grid = quantile_grid(n_quantiles=n_quantiles, eps=quantile_eps)
    Zq = samples_to_quantiles(X_list, u_grid)
    Yq = samples_to_quantiles(Y_list, u_grid)

    if bandwidth is None:
        bandwidth = bandwidth_scale * median_heuristic_bandwidth(Zq, u_grid)
        bandwidth = max(float(bandwidth), 1e-4)

    def delta(q1: np.ndarray, q2: np.ndarray) -> float:
        return w2_quantile_distance(q1, q2, u_grid)

    X_euclid = _coerce_covariates(c_list)

    model = PartiallyGlobalFrechet1D(
        bandwidth=bandwidth,
        delta=delta,
        ridge=ridge,
    )
    model.fit(X=X_euclid, Z=list(Zq), Y_quantiles=Yq, u_grid=u_grid)

    def predictor(X_new_samples: np.ndarray, c_new: Union[float, np.ndarray, Sequence[float]], n_pred: Optional[int] = None) -> np.ndarray:
        if n_pred is None:
            n_pred = len(X_new_samples)
        z_new_q = np.quantile(np.asarray(X_new_samples, dtype=float), u_grid)
        x_new = np.asarray(c_new, dtype=float)
        if x_new.ndim == 0:
            x_new = np.array([float(x_new)], dtype=float)
        else:
            x_new = x_new.reshape(-1)
        if x_new.shape[0] != X_euclid.shape[1]:
            raise ValueError(
                f"Covariate dimension mismatch: expected {X_euclid.shape[1]}, got {x_new.shape[0]}."
            )
        return model.predict_samples(x=x_new, z=z_new_q, n_samples=int(n_pred))

    info = {
        "model": model,
        "u_grid": u_grid,
        "Zq": Zq,
        "Yq": Yq,
        "bandwidth": bandwidth,
        "covariate_dim": X_euclid.shape[1],
    }
    return predictor, info
