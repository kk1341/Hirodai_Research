"""
Implementations of major algorithms discussed in
"Factor models for portfolio selection in large dimensions:
 the good, the better and the ugly"

The code is modular and research-oriented, intended for replication,
experimentation, and extension rather than black-box production use.

Algorithms covered:
1. Sample covariance ("the ugly")
2. Factor model covariance (known factors)
3. PCA / statistical factor model ("the good")
4. POET (Principal Orthogonal Complement Thresholding)
5. Ledoit–Wolf linear shrinkage
6. Nonlinear eigenvalue shrinkage (approximate)
7. GMV portfolio construction under each estimator

Dependencies: numpy, scipy, sklearn
"""

import numpy as np
from numpy.linalg import eigh, inv
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from scipy.stats import median_abs_deviation

# --------------------------------------------------
# Utilities
# --------------------------------------------------

def standardize_returns(R):
    """Demean returns"""
    return R - R.mean(axis=0)


def gmv_weights(Sigma):
    """Global Minimum Variance portfolio"""
    invS = inv(Sigma)
    ones = np.ones(Sigma.shape[0])
    w = invS @ ones
    return w / (ones @ w)


# --------------------------------------------------
# 1. Sample Covariance ("the ugly")
# --------------------------------------------------

def sample_covariance(R):
    R = standardize_returns(R)
    T = R.shape[0]
    return (R.T @ R) / T


# --------------------------------------------------
# 2. Known-factor model covariance
# --------------------------------------------------

def factor_covariance_known(R, F):
    """
    R: T x N returns
    F: T x K observed factors
    """
    R = standardize_returns(R)
    F = standardize_returns(F)

    B = np.linalg.lstsq(F, R, rcond=None)[0]  # K x N
    Sigma_f = np.cov(F, rowvar=False)
    U = R - F @ B
    Sigma_u = np.diag(np.var(U, axis=0))
    return B.T @ Sigma_f @ B + Sigma_u


# --------------------------------------------------
# 3. PCA / Statistical factor model ("the good")
# --------------------------------------------------

def pca_factor_covariance(R, K):
    R = standardize_returns(R)
    pca = PCA(n_components=K)
    F = pca.fit_transform(R)
    B = pca.components_.T
    Sigma_f = np.cov(F, rowvar=False)
    U = R - F @ B.T
    Sigma_u = np.diag(np.var(U, axis=0))
    return B @ Sigma_f @ B.T + Sigma_u


# --------------------------------------------------
# 4. POET estimator ("the better")
# --------------------------------------------------

def poet_covariance(R, K, threshold=None):
    """
    Simplified POET implementation
    """
    R = standardize_returns(R)
    T, N = R.shape
    S = sample_covariance(R)

    eigvals, eigvecs = eigh(S)
    idx = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    Lambda = eigvals[:K]
    V = eigvecs[:, :K]
    factor_part = V @ np.diag(Lambda) @ V.T

    residual = S - factor_part

    if threshold is None:
        threshold = np.sqrt(np.log(N) / T)

    R_th = residual.copy()
    for i in range(N):
        for j in range(N):
            if i != j and abs(R_th[i, j]) < threshold:
                R_th[i, j] = 0

    return factor_part + R_th


# --------------------------------------------------
# 5. Ledoit–Wolf Linear Shrinkage (Previously using sklearn, kept for reference)
# --------------------------------------------------

def ledoit_wolf_sklearn(R):
    lw = LedoitWolf().fit(R)
    return lw.covariance_


# --------------------------------------------------
# 6. Nonlinear eigenvalue shrinkage (approximate)
# --------------------------------------------------

def nonlinear_shrinkage_covariance(R, eps=1e-4):
    """
    Approximate nonlinear shrinkage via eigenvalue clipping
    (Pedagogical approximation)
    """
    S = sample_covariance(R)
    eigvals, eigvecs = eigh(S)

    eigvals_shrunk = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(eigvals_shrunk) @ eigvecs.T


# --------------------------------------------------
# 7. Linear Shrinkage to Identity (Custom Implementation)
# --------------------------------------------------

def linear_shrinkage_identity(X, assume_zero_mean=False):

    """
    Ledoit-Wolf線形収縮推定器（ターゲット行列はスケーリングされた単位行列 I）。

    Parameters
    ----------
    X : ndarray, shape (T, N)
        データ行列: T観測値 (行), N変数 (列).
    assume_zero_mean : bool
        Trueの場合、中心化をスキップ。Falseの場合、共分散推定前にデータを中心化する。

    Returns
    -------
    S : ndarray, shape (N, N)
        標本共分散行列 S.
    Sigma_hat : ndarray, shape (N, N)
        収縮共分散行列 c * mu * I + (1−c) * S.
    """
    T, N = X.shape

    # データの中心化
    if not assume_zero_mean:
        X = X - X.mean(axis=0, keepdims=True)

    # 標本共分散行列 S
    S = (1.0 / T) * (X.T @ X)

    # ターゲット行列の平均 mu_hat = (1/N) * Tr(S)
    mu_hat = (1.0 / N) * np.trace(S)

    # delta^2 の推定: || S − mu I ||^2_F
    S_minus = S - mu_hat * np.eye(N)
    delta2_hat = np.sum(S_minus * S_minus)

    # beta^2 の推定: E|| S − Sigma ||^2_F
    # ベクトル化による高速化 (ループ処理の排除)
    # X_outer: (T, N, N) - 各時点 t における x_t @ x_t.T
    X_outer = X[:, :, np.newaxis] * X[:, np.newaxis, :]
    diff = X_outer - S
    beta2_hat = np.sum(diff**2) / (T**2)

    # Shrinkage Intensity c_hat の計算
    if delta2_hat <= 0:
        c_hat = 0.0
    else:
        c_hat = beta2_hat / delta2_hat

    c_hat = np.clip(c_hat, 0.0, 1.0)  # 0 <= c_hat <= 1 にクリップ

    # 収縮共分散行列の構築
    Sigma_hat = c_hat * (mu_hat * np.eye(N)) + (1.0 - c_hat) * S

    return S, Sigma_hat


# --------------------------------------------------
# Example Usage
# --------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)
    T, N, K = 500, 50, 3

    R = np.random.randn(T, N) * 0.01
    F = np.random.randn(T, K)

    Sigma_sample = sample_covariance(R)
    Sigma_factor = factor_covariance_known(R, F)
    Sigma_pca = pca_factor_covariance(R, K)
    Sigma_poet = poet_covariance(R, K)
    
    # Custom shrinkage
    _, Sigma_custom_lw = linear_shrinkage_identity(R)
    
    Sigma_nls = nonlinear_shrinkage_covariance(R)

    w_gmv = gmv_weights(Sigma_poet)
    print("GMV weights (POET):", w_gmv[:5])