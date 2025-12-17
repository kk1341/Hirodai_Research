# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 11:50:15 2025

Compute the linear shrinkage estimator towards a multiple of the identity matrix
following Section 1 of Ledoit & Wolf (2022) "The Power of (Non-)Linear Shrinking".

@author: cting
"""

import numpy as np

def linear_shrinkage_identity(X, assume_zero_mean=False):
    """
    Parameters
    ----------
    X : ndarray, shape (T, N)
        Data matrix: T observations (rows), N variables (columns).
    assume_zero_mean : bool
        If False, the data will be demeaned (column‐wise) before covariance estimation.

    Returns
    -------
    Sigma_hat : ndarray, shape (N, N)
        The shrunk covariance matrix estimate: c * ℓ I + (1−c) * S.
    shrinkage : float
        The shrinkage intensity c ∈ [0,1].
    """

    T, N = X.shape

    if not assume_zero_mean:
        X = X - X.mean(axis=0, keepdims=True)

    # Sample covariance matrix
    
    S = (1.0 / T) * (X.T @ X)

    # Compute mu = (1/N) Tr(R) but we estimate Tr(R) by using Tr(S)
    
    mu_hat = (1.0 / N) * np.trace(S)

    # Compute delta^2 = || S − mu I ||^2_F
    # Note: Frobenius norm squared = sum of squares of entries
    # We use un‐scaled version (no division by N) 
    # but that just rescales all terms.
    
    S_minus = S - mu_hat * np.eye(N)
    delta2_hat = np.sum(S_minus * S_minus)

    # Estimate beta^2 = E|| S − Sigma ||^2_F
    # We replace with empirical estimate of variance of S
    # From Ledoit & Wolf (2004) style: 
    #    beta2_hat = (1/T) * sum_i sum_j Var(X_i X_j)
    # But we can estimate via: 
    #    beta2_hat = (1 / T**2) * [ sum_k || x_k x_k^T − S ||^2_F ]
    # where x_k is the k‐th observation (as column vector)
    # Equivalent: 
    #    beta2_hat = (1 / T^2) * sum_{t=1}^T || x_t x_t^T − S ||^2_F
    
    beta2_hat = 0.0
    for t in range(T):
        xt = X[t, :].reshape(-1,1)
        outer = (xt @ xt.T)
        diff = outer - S
        beta2_hat += np.sum(diff * diff)
    beta2_hat /= (T**2)

    # Then a^2_T = d^2_T − b^2_T
    alpha2_hat = max(delta2_hat - beta2_hat, 0)  # ensure non‐negative

    # Compute shrinkage intensity c_hat = beta2_hat / delta2_hat
    # If delta2_hat = 0 (means S = ℓ I exactly) then shrinkage = 0
    if delta2_hat <= 0:
        c_hat = 0.0
    else:
        c_hat = beta2_hat / delta2_hat
    c_hat = min(max(c_hat, 0.0), 1.0)

    # Construct shrunk covariance
    Sigma_hat = c_hat * (mu_hat * np.eye(N)) + (1.0 - c_hat) * S

    return Sigma_hat, c_hat


if __name__ == "__main__":
    np.random.seed(137)
    
    # Generate data: T=30 observations, N=5 variables
    T, N = 30, 5
    
    # True covariance = random positive‐definite
    A = np.random.randn(N, N)
    true_cov = A @ A.T
    
    # Generate zero‐mean multivariate Gaussian
    X = np.random.multivariate_normal(np.zeros(N), true_cov, size=T)
    X = X - X.mean(axis=0, keepdims=True)
    
    # Sample covariance matrix
    S = (1.0 / T) * (X.T @ X)

    # Compute mu = (1/N) Tr(R) but we estimate Tr(R) by using Tr(S)
    mu_hat = (1.0 / N) * np.trace(S)

    # Compute delta^2 = || S − mu I ||^2_F
    # Note: Frobenius norm squared = sum of squares of entries
    # We use un‐scaled version (no division by N) 
    # but that just rescales all terms.
    
    S_minus = S - mu_hat * np.eye(N)
    delta2_hat = np.sum(S_minus * S_minus)

    # Estimate beta^2 = E|| S − Sigma ||^2_F
    # We replace with empirical estimate of variance of S
    # From Ledoit & Wolf (2004) style: 
    #    beta2_hat = (1/T) * sum_i sum_j Var(X_i X_j)
    # But we can estimate via: 
    #    beta2_hat = (1 / T**2) * [ sum_k || x_k x_k^T − S ||^2_F ]
    # where x_k is the k‐th observation (as column vector)
    # Equivalent: 
    #    beta2_hat = (1 / T^2) * sum_{t=1}^T || x_t x_t^T − S ||^2_F
    
    beta2_hat = 0.0
    for t in range(T):
        xt = X[t, :].reshape(-1,1)
        outer = (xt @ xt.T)
        diff = outer - S
        beta2_hat += np.sum(diff * diff)
    beta2_hat /= (T**2)

    # Then a^2_T = d^2_T − b^2_T
    alpha2_hat = max(delta2_hat - beta2_hat, 0)  # ensure non‐negative


    # Compute shrinkage intensity c_hat = beta2_hat / delta2_hat
    # If delta2_hat = 0 (means S = mu I exactly) then shrinkage = 0
    if delta2_hat <= 0:
        c_hat = 0.0
    else:
        c_hat = beta2_hat / delta2_hat
    c_hat = min(max(c_hat, 0.0), 1.0)

    # Construct shrunk covariance
    Sigma_hat = c_hat * (mu_hat * np.eye(N)) + (1.0 - c_hat) * S

    print(Sigma_hat)
    print(c_hat)


















#    Sigma_shrunk, shrinkage_intensity = linear_shrinkage_identity\
#                                        (X, assume_zero_mean=False)
#    print("Shrinkage intensity c_hat:", shrinkage_intensity)
#    print("Shrunk covariance matrix shape:", Sigma_shrunk.shape)
