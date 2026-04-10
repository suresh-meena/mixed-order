from __future__ import annotations

import os
import numpy as np

from experiments.plot_helpers import apply_pub_style, save_fig, plt

RESULT_DIR = os.path.dirname(__file__)


def main() -> None:
    apply_pub_style()
    data = np.load(os.path.join(RESULT_DIR, "limiting_cases_results.npz"), allow_pickle=True)

    p_values = data["p_values"]
    th_pw = data["th_pw"]
    th_3b = data["th_3b"]
    emp_pw = data["emp_pw"]
    emp_3b = data["emp_3b"]
    lam_sweep = data["lam_sweep"]
    pc_th_sw = data["pc_th_sw"]
    pc_emp_sw = data["pc_emp_sw"]
    p_fixed = float(data["p_fixed"])
    lam_opt_fixed = float(data["lam_opt_fixed"])
    N = int(data["N"])
    beta = float(data["beta"])
    alpha_c = 0.138

    fig1, ax1 = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    p_dense = np.linspace(min(p_values) * 0.95, max(p_values) * 1.05, 300)
    qt_dense = 6.0 * beta - 3.0 * p_dense
    ax1.plot(p_dense, alpha_c * p_dense * N, "--", color="#0077BB", label=r"Theory: $\alpha_c p N$")
    ax1.scatter(p_values, emp_pw, c="#0077BB", s=60, label=r"Empirical ($\lambda\approx 0$)")
    ax1.plot(p_dense, alpha_c * qt_dense * N / 2, "--", color="#CC3311", label=r"Theory: $\alpha_c\tilde{q}N/2$")
    ax1.scatter(p_values, emp_3b, c="#CC3311", s=60, marker="s", label=r"Empirical ($\lambda\approx\infty$)")
    ax1.set_xlabel(r"Edge density $p$")
    ax1.set_ylabel(r"Critical capacity $P_c$")
    ax1.set_title(rf"Limiting cases — pairwise vs. 3-body  ($N={N},\;\beta={beta}$)")
    ax1.legend(loc="upper left", ncol=2)
    path1 = os.path.join(RESULT_DIR, "limiting_cases_p_sweep.png")
    fig1.savefig(path1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax2.plot(lam_sweep, pc_th_sw, "-", color="#222222", label="Replica theory")
    ax2.plot(lam_sweep, pc_emp_sw, "o-", color="#EE7733", label="Simulation")
    ax2.axvline(lam_opt_fixed, color="#009988", ls="--", label=rf"$\lambda^*={lam_opt_fixed:.2f}$")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\lambda = b/a$")
    ax2.set_ylabel(r"Critical capacity $P_c$")
    ax2.set_title(rf"Capacity vs. $\lambda$  ($p={p_fixed},\;N={N},\;\beta={beta}$)")
    ax2.legend(loc="center right")
    path2 = os.path.join(RESULT_DIR, "limiting_cases_lambda_sweep.png")
    fig2.savefig(path2)
    plt.close(fig2)
    print(f"saved plots to {path1} and {path2}")


if __name__ == "__main__":
    main()