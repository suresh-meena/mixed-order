import numpy as np

def compute_q_from_budget(p, N, beta=0.5):
    M = beta * N ** 2
    M2 = 0.5 * p * N ** 2
    M3_needed = M - M2
    q = max(6.0 * M3_needed / (N ** 3), 0.0)
    q_tilde = q * N
    return q, q_tilde

def replica_capacity(p, N, beta=0.5, alpha_c=0.138, lam=2.0):
    q_tilde = 6.0 * beta - 3.0 * p
    if q_tilde <= 0 or p <= 0:
        return 0.0
    signal_sq = (1.0 + lam / 2.0) ** 2
    noise_inv = 1.0 / p + lam ** 2 / (2.0 * q_tilde)
    return alpha_c * N * signal_sq / noise_inv

def optimal_lambda(p, beta=0.5):
    q_tilde = 6.0 * beta - 3.0 * p
    if p <= 0 or q_tilde <= 0:
        return np.inf
    return q_tilde / p

def capacity_on_optimal_line(p, N, beta=0.5, alpha_c=0.138):
    q_tilde = 6.0 * beta - 3.0 * p
    if q_tilde <= 0 or p <= 0:
        return 0.0
    return alpha_c * N * (6.0 * beta - p) / 2.0

def replica_capacity_structured(p, N, beta=0.5, g2=0.0, alpha_c=0.138, lam=2.0):
    q_tilde = 6.0 * beta - 3.0 * p
    if q_tilde <= 0 or p <= 0:
        return 0.0
    signal_sq = (1.0 + lam / 2.0) ** 2
    noise_inv = (1.0 / p) + g2 + (lam ** 2) / (2.0 * q_tilde)
    return alpha_c * N * signal_sq / noise_inv

def optimal_lambda_structured(p, beta=0.5, g2=0.0):
    q_tilde = 6.0 * beta - 3.0 * p
    if p <= 0 or q_tilde <= 0:
        return np.inf
    return q_tilde * (1.0 / p + g2)
