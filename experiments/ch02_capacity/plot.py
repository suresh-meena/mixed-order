from __future__ import annotations

from pathlib import Path
import numpy as np

from experiments.common import apply_pub_style, get_color_palette, load_npz, plt, save_fig
from experiments.plot_helpers import edges_from_centers

RESULT_DIR = Path(__file__).resolve().parent


def main() -> None:
    apply_pub_style()
    cp = get_color_palette()
    data = load_npz(RESULT_DIR / "ch02_results.npz")

    N = int(data["N"][0])
    p_vals = data["p_vals"]
    lam_vals = data["lam_vals"]
    pc_grid = data["pc_grid"]
    lam_opt_curve = data["lam_opt_curve"]
    conn_cutoff = float(data["connectivity_cutoff"][0])

    p_edges = edges_from_centers(p_vals)
    lam_edges = edges_from_centers(lam_vals)

    # Panel A: heatmap of critical capacity (single plot)
    fig_h, ax_h = plt.subplots(figsize=(8.0, 6.0))
    im = ax_h.pcolormesh(p_edges, lam_edges, pc_grid, cmap="plasma", shading="auto")
    ax_h.plot(p_vals, lam_opt_curve, "w--", lw=1.8, label=r"$\lambda^*(p)$")
    ax_h.axvline(conn_cutoff, color=cp["uncentered"], ls=":", lw=2.0, label=r"$\log N/N$")
    ax_h.set_yscale("log")
    ax_h.set_xlim(float(p_vals.min()), float(p_vals.max()))
    ax_h.set_ylim(float(lam_vals.min()), float(lam_vals.max()))
    ax_h.set_xlabel("p")
    ax_h.set_ylabel(r"$\lambda$")
    ax_h.set_title(f"Chapter 2.1 heatmap ($N={N}$)")
    ax_h.legend(frameon=False)
    fig_h.colorbar(im, ax=ax_h, label=r"$P_c$")
    save_fig(fig_h, RESULT_DIR / "ch02_capacity_heatmap.png")
    plt.close(fig_h)

    # Panel B: representative Pc slices and Pc on optimal lambda curve
    fig_s, ax_s = plt.subplots(figsize=(8.0, 5.0))
    pick = data["lambda_pick_idx"].astype(int)
    pc_lines = data["pc_lines"]
    for row, idx in enumerate(pick.tolist()):
        ax_s.plot(p_vals, pc_lines[row], "o-", label=rf"$\lambda={lam_vals[idx]:.2f}$")
    lam_log = np.log(lam_vals)
    pc_star = np.zeros_like(p_vals, dtype=float)
    for j, p in enumerate(p_vals.tolist()):
        lam_star = lam_opt_curve[j]
        lam_star = float(np.clip(lam_star, lam_vals.min(), lam_vals.max()))
        pc_star[j] = float(np.interp(np.log(lam_star), lam_log, pc_grid[:, j]))
    ax_s.plot(p_vals, pc_star, "k--", lw=2.2, label=r"$P_c$ on $\lambda^*(p)$")
    ax_s.axvline(conn_cutoff, color=cp["uncentered"], ls=":", lw=2.0)
    ax_s.set_xlim(float(p_vals.min()), float(p_vals.max()))
    ax_s.set_xlabel("p")
    ax_s.set_ylabel(r"$P_c$")
    ax_s.set_title("Chapter 2.1 slices")
    ax_s.legend(frameon=False)
    save_fig(fig_s, RESULT_DIR / "ch02_capacity_slices.png")
    plt.close(fig_s)

    sn_p = data["sn_p_grid"].astype(float)
    pairs_p = data["sn_pairs_p"]
    pairs_lam = data["sn_pairs_lam"]
    sig_emp = data["sn_signal_emp"]
    sig_th = data["sn_signal_theory"]
    tau_emp = data["sn_tau2_emp"]
    tau_th = data["sn_tau2_theory"]

    # Panel: signal vs P (single plot)
    fig_sig, ax_sig = plt.subplots(figsize=(7.6, 5.0))
    for k in range(pairs_p.size):
        lbl = rf"$p={pairs_p[k]:.2f},\,\lambda={pairs_lam[k]:.1f}$"
        ax_sig.plot(sn_p, sig_emp[k], "o-", label=f"emp {lbl}")
        ax_sig.plot(sn_p, sig_th[k], "--", color=ax_sig.lines[-1].get_color())
    ax_sig.set_xlabel("P")
    ax_sig.set_ylabel("Signal")
    ax_sig.set_title("Chapter 2.2 signal")
    ax_sig.legend(frameon=False, fontsize=9)
    save_fig(fig_sig, RESULT_DIR / "ch02_signal.png")
    plt.close(fig_sig)

    # Panel: crosstalk variance vs P (single plot)
    fig_tau, ax_tau = plt.subplots(figsize=(7.6, 5.0))
    for k in range(pairs_p.size):
        lbl = rf"$p={pairs_p[k]:.2f},\,\lambda={pairs_lam[k]:.1f}$"
        ax_tau.plot(sn_p, tau_emp[k], "o-", label=f"emp {lbl}")
        ax_tau.plot(sn_p, tau_th[k], "--", color=ax_tau.lines[-1].get_color())
    ax_tau.set_xlabel("P")
    ax_tau.set_ylabel(r"Crosstalk variance $\tau^2$")
    ax_tau.set_title("Chapter 2.2 crosstalk")
    save_fig(fig_tau, RESULT_DIR / "ch02_tau2.png")
    plt.close(fig_tau)

    gp = data["gram_p_grid"].astype(float)
    diag = data["gram_diag_mass"]
    off = data["gram_offdiag_var"]
    ratio = data["gram_ratio"]

    # Panel: Gram diagonal mass and off-diagonal variance (single plot)
    fig_diag, ax_diag = plt.subplots(figsize=(8.0, 5.0))
    ax_diag.plot(gp, diag, "o-", color=cp["centered"], label="diag mass")
    ax_diag.plot(gp, off, "s--", color=cp["extra"], label="offdiag var")
    ax_diag.set_xlabel("P")
    ax_diag.set_ylabel("Mass / variance")
    ax_diag.set_title("Chapter 2.3 Gram diagnostics")
    ax_diag.legend(frameon=False)
    save_fig(fig_diag, RESULT_DIR / "ch02_gram_diag_off.png")
    plt.close(fig_diag)

    # Panel: diagonal / off-diagonal ratio (single plot)
    fig_ratio, ax_ratio = plt.subplots(figsize=(8.0, 5.0))
    ax_ratio.plot(gp, ratio, "o-", color=cp["mixed"])
    ax_ratio.set_xlabel("P")
    ax_ratio.set_ylabel("diag/offdiag")
    ax_ratio.set_title("Diagonal dominance ratio")
    save_fig(fig_ratio, RESULT_DIR / "ch02_gram_ratio.png")
    plt.close(fig_ratio)


if __name__ == "__main__":
    main()
