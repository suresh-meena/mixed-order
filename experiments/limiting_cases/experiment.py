import os
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from experiments.plot_helpers import apply_pub_style
from mixed_order.theory import (
    compute_q_from_budget,
    replica_capacity,
    optimal_lambda,
    capacity_on_optimal_line,
)
from mixed_order.metrics import (
    find_empirical_pc,
    run_batched_retrieval,
    seed_all,
    _DEVICE,
)

RESULT_DIR = os.path.dirname(__file__)

apply_pub_style()


def limiting_cases(N, beta, n_trials, n_seeds, n_p_values=8, n_lam_sweep=25):
    alpha_c  = 0.138
    p_values = np.linspace(0.15, 0.85, n_p_values)
    lam_pw   = 0.01
    lam_3b   = 100.0
    p_fixed  = 0.35

    # ── p sweep ─────────────────────────────────────────────────────────────
    th_pw, th_3b, emp_pw, emp_3b = [], [], [], []
    with tqdm(p_values, desc="  p sweep", unit="p", ncols=90) as pbar:
        for p in pbar:
            _, qt = compute_q_from_budget(p, N, beta)
            pc_pw_th = alpha_c * p  * N
            pc_3b_th = alpha_c * qt * N / 2.0
            pc_pw = find_empirical_pc(N, p, beta, lam_pw, n_trials, n_seeds, pbar=pbar)
            pc_3b = find_empirical_pc(N, p, beta, lam_3b, n_trials, n_seeds, pbar=pbar)
            th_pw.append(pc_pw_th);  th_3b.append(pc_3b_th)
            emp_pw.append(pc_pw);    emp_3b.append(pc_3b)

    th_pw  = np.array(th_pw);  th_3b  = np.array(th_3b)
    emp_pw = np.array(emp_pw, float);  emp_3b = np.array(emp_3b, float)
    rho_pw, _ = spearmanr(th_pw, emp_pw)
    rho_3b, _ = spearmanr(th_3b, emp_3b)

    # ── lambda sweep ────────────────────────────────────────────────────────
    _, qt_fixed = compute_q_from_budget(p_fixed, N, beta)
    lam_sweep   = np.logspace(-2, 2, n_lam_sweep)
    pc_th_sw, pc_emp_sw = [], []
    with tqdm(lam_sweep, desc=f"  lam sweep p={p_fixed}", unit="lam", ncols=90) as pbar:
        for lam in pbar:
            pc_th_sw.append(replica_capacity(p_fixed, N, beta, lam=lam))
            pc_emp_sw.append(find_empirical_pc(N, p_fixed, beta, lam, n_trials, n_seeds, pbar=pbar))

    np.savez(
        os.path.join(RESULT_DIR, "limiting_cases_results.npz"),
        N=N,
        beta=beta,
        n_trials=n_trials,
        n_seeds=n_seeds,
        n_p_values=n_p_values,
        n_lam_sweep=n_lam_sweep,
        p_values=p_values,
        th_pw=th_pw,
        th_3b=th_3b,
        emp_pw=emp_pw,
        emp_3b=emp_3b,
        lam_sweep=lam_sweep,
        pc_th_sw=pc_th_sw,
        pc_emp_sw=pc_emp_sw,
        p_fixed=p_fixed,
        lam_pw=lam_pw,
        lam_3b=lam_3b,
        rho_pw=rho_pw,
        rho_3b=rho_3b,
    )

    # ── Plotting ────────────────────────────────────────────────────────────
    _C_PW  = "#0077BB"
    _C_3B  = "#CC3311"
    _C_OPT = "#009988"
    p_dense = np.linspace(min(p_values) * 0.95, max(p_values) * 1.05, 300)
    qt_dense = 6.0 * beta - 3.0 * p_dense
    lam_opt_fixed = optimal_lambda(p_fixed, beta)
    pc_opt_fixed  = capacity_on_optimal_line(p_fixed, N, beta)

    # Fig 1: p sweep
    fig1, ax1 = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax1.plot(p_dense, alpha_c * p_dense * N, "--", color=_C_PW, label=r"Theory: $\alpha_c p N$")
    ax1.scatter(p_values, emp_pw, c=_C_PW, s=60, label=r"Empirical ($\lambda\approx 0$)")
    ax1.plot(p_dense, alpha_c * qt_dense * N / 2, "--", color=_C_3B, label=r"Theory: $\alpha_c\tilde{q}N/2$")
    ax1.scatter(p_values, emp_3b, c=_C_3B, s=60, marker="s", label=r"Empirical ($\lambda\approx\infty$)")
    ax1.set_xlabel(r"Edge density $p$")
    ax1.set_ylabel(r"Critical capacity $P_c$")
    ax1.set_title(rf"Limiting cases — pairwise vs. 3-body  ($N={N},\;\beta={beta}$)")
    ax1.legend(loc="upper left", ncol=2)
    path1 = os.path.join(RESULT_DIR, "limiting_cases_p_sweep.png")
    fig1.savefig(path1)
    plt.close(fig1)

    # Fig 2: lambda sweep
    fig2, ax2 = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax2.plot(lam_sweep, pc_th_sw,  "-",  color="#222222", label="Replica theory")
    ax2.plot(lam_sweep, pc_emp_sw, "o-", color="#EE7733", label="Simulation")
    ax2.axvline(lam_opt_fixed, color=_C_OPT, ls="--", label=rf"$\lambda^*={lam_opt_fixed:.2f}$")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\lambda = b/a$")
    ax2.set_ylabel(r"Critical capacity $P_c$")
    ax2.set_title(rf"Capacity vs. $\lambda$  ($p={p_fixed},\;N={N},\;\beta={beta}$)")
    ax2.legend(loc="center right")
    path2 = os.path.join(RESULT_DIR, "limiting_cases_lambda_sweep.png")
    fig2.savefig(path2)
    plt.close(fig2)
    print(f"  Saved {path1} and {path2}")