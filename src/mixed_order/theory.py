import numpy as np

def compute_q_from_budget(p, N, beta=0.5):
    # Scalar only
    M = beta * N ** 2
    M2 = 0.5 * p * N ** 2
    M3_needed = M - M2
    q = max(6.0 * M3_needed / (N ** 3), 0.0)
    q_tilde = q * N
    return q, q_tilde

def replica_capacity(p, N, beta=0.5, alpha_c=0.138, lam=2.0):
    # Vectorized
    q_tilde = 6.0 * beta - 3.0 * p
    signal_sq = (1.0 + lam / 2.0) ** 2
    noise_inv = 1.0 / p + lam ** 2 / (2.0 * q_tilde)
    res = alpha_c * N * signal_sq / noise_inv
    # Mask invalid regions
    invalid = (q_tilde <= 0) | (p <= 0)
    if np.isscalar(res):
        return 0.0 if invalid else float(res)
    res[invalid] = 0.0
    return res

def optimal_lambda(p, beta=0.5):
    # Vectorized
    q_tilde = 6.0 * beta - 3.0 * p
    invalid = (p <= 0) | (q_tilde <= 0)
    res = q_tilde / p
    if np.isscalar(res):
        return np.inf if invalid else float(res)
    res[invalid] = np.inf
    return res

def capacity_on_optimal_line(p, N, beta=0.5, alpha_c=0.138):
    # Vectorized
    q_tilde = 6.0 * beta - 3.0 * p
    res = alpha_c * N * (6.0 * beta - p) / 2.0
    invalid = (q_tilde <= 0) | (p <= 0)
    if np.isscalar(res):
        return 0.0 if invalid else float(res)
    res[invalid] = 0.0
    return res

def replica_capacity_structured(p, N, beta=0.5, g2=0.0, alpha_c=0.138, lam=2.0):
    # Vectorized
    q_tilde = 6.0 * beta - 3.0 * p
    signal_sq = (1.0 + lam / 2.0) ** 2
    noise_inv = (1.0 / p) + g2 + (lam ** 2) / (2.0 * q_tilde)
    res = alpha_c * N * signal_sq / noise_inv
    invalid = (q_tilde <= 0) | (p <= 0)
    if np.isscalar(res):
        return 0.0 if invalid else float(res)
    res[invalid] = 0.0
    return res

def optimal_lambda_structured(p, beta=0.5, g2=0.0):
    # Vectorized
    q_tilde = 6.0 * beta - 3.0 * p
    res = q_tilde * (1.0 / p + g2)
    invalid = (p <= 0) | (q_tilde <= 0)
    if np.isscalar(res):
        return np.inf if invalid else float(res)
    res[invalid] = np.inf
    return res
