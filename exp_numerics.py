"""
Numerical experiments for the SHN thesis (PyTorch version).

Produces two figures:
  plots/heatmap_p_lambda.png  – capacity heatmap in (p, lambda)
    plots/limiting_cases_p_sweep.png      – limiting-case p-sweep
    plots/limiting_cases_lambda_sweep.png – limiting-case lambda-sweep
"""

import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from shn_model import SimplicialHopfieldNetwork

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Publication-quality matplotlib style
# ─────────────────────────────────────────────────────────────────────────────

def _apply_pub_style():
    """Set global rcParams for a clean, thesis-ready look (Palatino + Euler VM)."""
    import shutil, subprocess

    def _pkg_available(sty: str) -> bool:
        """Return True if a LaTeX .sty file is locatable via kpsewhich."""
        if shutil.which("kpsewhich") is None:
            return False
        r = subprocess.run(["kpsewhich", sty],
                           capture_output=True, text=True)
        return bool(r.stdout.strip())

    have_latex    = shutil.which("latex") is not None
    have_dvipng   = shutil.which("dvipng") is not None
    have_mathpazo = _pkg_available("mathpazo.sty")
    have_eulervm  = _pkg_available("eulervm.sty")
    have_type1cm  = _pkg_available("type1cm.sty")

    if have_latex and have_dvipng and have_type1cm and (have_mathpazo or have_eulervm):
        preamble = ""
        if have_mathpazo:
            preamble += r"\usepackage{mathpazo}"   # Palatino text + Pazo math
        if have_eulervm:
            preamble += r"\usepackage{eulervm}"    # Euler VM overlaid for math
        if not have_eulervm:
            print("  [style] eulervm not found — using mathpazo only.\n"
                  "  Install with: sudo dnf install texlive-eulervm")
        plt.rcParams["text.usetex"]         = True
        plt.rcParams["text.latex.preamble"] = preamble
    elif have_mathpazo or have_eulervm:
        # Incomplete TeX install (common on minimal distros): avoid savefig crash.
        plt.rcParams["text.usetex"] = False
        plt.rcParams["mathtext.fontset"] = "cm"
        print("  [style] LaTeX packages found but TeX prerequisites are incomplete.\n"
              "  Falling back to mathtext (install texlive-type1cm and dvipng to enable usetex).")
    else:
        # Pure-Python fallback: matplotlib's built-in CM-like fonts
        plt.rcParams["mathtext.fontset"] = "cm"
        print("  [style] LaTeX/mathpazo unavailable — falling back to mathtext.")

    plt.rcParams.update({
        "font.family":        "serif",
        "font.size":          13,
        "axes.titlesize":     14,
        "axes.labelsize":     13,
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "legend.fontsize":    11,
        "figure.titlesize":   15,
        "lines.linewidth":    2.0,
        "lines.markersize":   6,
        "axes.linewidth":     1.0,
        "xtick.major.width":  1.0,
        "ytick.major.width":  1.0,
        "xtick.minor.width":  0.6,
        "ytick.minor.width":  0.6,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.grid":          True,
        "grid.alpha":         0.35,
        "grid.linewidth":     0.6,
        "grid.linestyle":     "--",
        "figure.dpi":         150,
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
    })

_apply_pub_style()

# ─────────────────────────────────────────────────────────────────────────────
# Analytical formulas
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# Core: empirical capacity search
# ─────────────────────────────────────────────────────────────────────────────

def _seed_all(seed: int):
    """Seed both PyTorch and NumPy global RNGs from a single integer."""
    torch.manual_seed(seed)
    np.random.seed(int(seed) & 0xFFFF_FFFF)


def run_retrieval_experiment(N, P, p, beta=0.5, n_trials=5,
                             max_steps=100, seed=42, lam=2.0,
                             device=_DEVICE, topology_cache=None):
    """
    Run a retrieval experiment and return the mean overlap with the target pattern.

    Seeding strategy (fully reproducible):
      * Topology : seeded with ``seed``            (shared across lam values)
      * Patterns : seeded with ``seed ^ 0xABCDEF``  (decoupled from topology)
      * Trial k  : seeded with ``seed * 10007 + k``  (per-trial independence)
    Both torch and numpy RNGs are seeded consistently via _seed_all().
    """
    q, q_tilde = compute_q_from_budget(p, N, beta)
    if q_tilde < 1e-10 and p < 1e-10:
        return 0.0  # degenerate case

    model = SimplicialHopfieldNetwork(N, p, beta, lam, device=device)

    # ── Topology (seed fixed; cached across lam at same seed) ─────────────────
    if topology_cache is not None:
        cache_key = (int(N), float(p), float(beta), int(seed))
        cached = topology_cache.get(cache_key)
        if cached is None:
            _seed_all(seed)
            model.generate_masks()
            topology_cache[cache_key] = (
                model.c, model.tri_i, model.tri_j, model.tri_k,
                model.n_tri, model.tri_adj_indices, model.tri_adj_indptr,
                model.all_tri_idx,
            )
        else:
            (model.c, model.tri_i, model.tri_j, model.tri_k,
             model.n_tri, model.tri_adj_indices, model.tri_adj_indptr,
             model.all_tri_idx) = cached
    else:
        _seed_all(seed)
        model.generate_masks()

    # ── Patterns (decoupled from topology seed) ────────────────────────────────
    _seed_all(seed ^ 0xABCDEF)
    patterns = torch.randint(0, 2, (P, N), device=device) * 2 - 1
    model.store_patterns(patterns)

    # ── Build all initial states in one batch and run dynamics together ────────
    # Per-trial seed governs only the flip-index noise; patterns are fixed above.
    inits   = torch.empty(n_trials, N, device=device)
    targets = torch.empty(n_trials, N, device=device)
    for trial in range(n_trials):
        _seed_all(seed * 10007 + trial)
        mu     = trial % P
        target = patterns[mu].float()
        n_flip = max(1, N // 10)
        flip_idx = torch.randperm(N, device=device)[:n_flip]
        init = target.clone()
        init[flip_idx] *= -1
        inits[trial]   = init
        targets[trial] = target

    # Single batched forward pass instead of n_trials sequential runs
    finals = model.run(inits, max_steps=max_steps)       # (n_trials, N)

    overlaps = (finals * targets).sum(dim=1) / N         # (n_trials,)
    return float(overlaps.mean().item())

def find_empirical_pc(N, p, beta, lam, n_trials, n_seeds,
                      threshold=0.99, pbar=None):
    """
    Return fractional P_c: interpolated between the last P that fully passes
    and the first P that fails, weighted by the pass fraction at the boundary.
    This avoids integer quantisation steps in plots.
    """
    max_p = int(min(N, max(64, int(0.8 * N))))
    # Step size scales with N so the sweep stays O(log N) for large N.
    step  = max(1, N // 20)
    p_candidates = np.unique(np.concatenate([
        np.arange(1, 15),
        np.arange(15, max_p + 1, step),
    ])).astype(int)

    topology_cache = {}
    # Memoize completed (full-seed) overlap evaluations to avoid redundant calls
    # at the interpolation step.
    _memo: dict = {}

    def _mean_overlap(P, stop_on_fail=False):
        """Run all seeds and return mean overlap.

        When ``stop_on_fail`` is True, exits early as soon as the remaining
        seeds cannot raise the average above ``threshold`` — used during the
        coarse/fine sweep for speed.  Results from a full run are cached.
        """
        P = int(P)
        if P in _memo:
            return _memo[P]
        total = 0.0
        for seed in range(n_seeds):
            m = run_retrieval_experiment(
                N, P, p, beta,
                n_trials=n_trials, seed=seed, lam=lam,
                device=_DEVICE,
                topology_cache=topology_cache,
            )
            total += m
            if pbar is not None:
                pbar.set_postfix(
                    p=f"{p:.2f}", lam=f"{lam:.1f}",
                    P=P, seed=f"{seed + 1}/{n_seeds}",
                    refresh=False,
                )
            # Early-fail shortcut: remaining seeds can't push average to threshold
            if stop_on_fail:
                remaining = n_seeds - seed - 1
                if (total + remaining) / n_seeds < threshold:
                    return total / (seed + 1)  # underestimate — do NOT cache
        result = total / n_seeds
        _memo[P] = result
        return result

    def _passes_capacity(P):
        return _mean_overlap(P, stop_on_fail=True) >= threshold

    # Coarse sweep to bracket the transition
    best = 0
    first_fail = None
    for P in p_candidates:
        if _passes_capacity(P):
            best = int(P)
        else:
            first_fail = int(P)
            break

    if first_fail is None:
        return float(best)

    # Fine sweep to resolve to integer boundary
    for P in range(best + 1, first_fail):
        if _passes_capacity(P):
            best = int(P)
        else:
            first_fail = int(P)
            break

    # Fractional interpolation using memoized (accurate) values.
    # _mean_overlap without stop_on_fail to guarantee full-seed accuracy.
    m_pass = _mean_overlap(best)       if best > 0 else 1.0
    m_fail = _mean_overlap(first_fail)
    denom  = m_pass - m_fail
    if denom > 1e-6:
        frac = (m_pass - threshold) / denom
        return best + frac * (first_fail - best)
    return float(best)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment A – capacity heatmap in (p, lambda)
# ─────────────────────────────────────────────────────────────────────────────

def heatmap_p_lambda(N, beta, n_p, n_lam, n_trials, n_seeds):
    alpha_c  = 0.138
    p_vals   = np.linspace(0.10, 0.90, n_p)
    lam_vals = np.logspace(np.log10(0.50), np.log10(5.00), n_lam)

    # Analytical prediction
    p_g, lam_g = np.meshgrid(p_vals, lam_vals)
    q_g        = 6.0 * beta - 3.0 * p_g
    valid      = (p_g > 0) & (q_g > 0)
    pc_anal    = np.where(
        valid,
        alpha_c * N * (1 + lam_g / 2) ** 2 / (1 / p_g + lam_g ** 2 / (2 * q_g)),
        0.0,
    )

    # Empirical grid
    pc_emp = np.zeros_like(pc_anal)
    cells  = [
        (i, j, float(lam), float(p))
        for i, lam in enumerate(lam_vals)
        for j, p   in enumerate(p_vals)
    ]

    with tqdm(cells, desc="  Heatmap", unit="cell", ncols=90) as pbar:
        for i, j, lam, p in pbar:
            _, qt = compute_q_from_budget(p, N, beta)
            if qt < 0.01:
                continue
            pc_emp[i, j] = find_empirical_pc(
                N, p, beta, lam, n_trials, n_seeds, pbar=pbar,
            )
            pbar.set_postfix(
                p=f"{p:.2f}", lam=f"{lam:.3g}", Pc=int(pc_emp[i, j]),
                refresh=True,
            )

    # ── Build plot arrays ─────────────────────────────────────────────────────
    p_line   = np.linspace(0.05, 0.95, 300)
    lam_opt  = np.array([optimal_lambda(pp, beta) for pp in p_line])
    in_range = (lam_opt >= lam_vals.min()) & (lam_opt <= lam_vals.max())

    def _edges_from_centers(vals):
        vals    = np.asarray(vals, dtype=float)
        edges   = np.empty(vals.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (vals[:-1] + vals[1:])
        edges[0]    = vals[0]  - 0.5 * (vals[1]  - vals[0])
        edges[-1]   = vals[-1] + 0.5 * (vals[-1] - vals[-2])
        return edges

    p_edges   = _edges_from_centers(p_vals)
    lam_edges = _edges_from_centers(lam_vals)

    # Shared colour scale so both panels are directly comparable
    vmax = max(float(pc_anal.max()), float(pc_emp.max()))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True, squeeze=False)
    ax_theory, ax_emp = axes[0]

    panel_info = [
        (ax_theory, pc_anal, "(a) Analytical replica theory"),
        (ax_emp, pc_emp,  "(b) Empirical simulation"),
    ]
    for ax, data, title in panel_info:
        im = ax.pcolormesh(
            p_edges, lam_edges, data,
            cmap="plasma", shading="auto",
            vmin=0, vmax=vmax,
        )
        # Optimal-lambda ridge shown as a dashed white line
        ax.plot(p_line[in_range], lam_opt[in_range],
                color="white", lw=2.0, ls="--",
                label=r"$\lambda^*(p)=\tilde{q}/p$")
        ax.set_yscale("log")
        ax.set_ylim(lam_vals[0], lam_vals[-1])
        ax.set_xlim(p_edges[0],  p_edges[-1])
        ax.set_xlabel(r"Edge density $p$")
        ax.set_ylabel(r"$\lambda = b/a$")
        ax.set_title(title, pad=7)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"${y:g}$"))
        ax.legend(loc="upper right", framealpha=0.75,
                  labelcolor="white", facecolor="#333333",
                  edgecolor="none", fontsize=10)
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(r"$P_c$  (critical capacity)", labelpad=8)

    # Annotate mean absolute relative error on the empirical panel
    diff_pct = 100.0 * (pc_emp - pc_anal) / np.where(pc_anal > 0, pc_anal, np.nan)
    mae_pct  = float(np.nanmean(np.abs(diff_pct)))
    ax_emp.text(
        0.03, 0.04,
        f"MAE vs. theory: {mae_pct:.1f}%",
        transform=ax_emp.transAxes, fontsize=10, color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="#00000099", ec="none"),
    )

    fig.suptitle(
        rf"SHN capacity heatmap  ($N={N},\;\beta={beta},\;\mathrm{{seeds}}={n_seeds}$)",
    )

    path = os.path.join(PLOT_DIR, "heatmap_p_lambda.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Experiment B – limiting cases
# ─────────────────────────────────────────────────────────────────────────────

def limiting_cases(N, beta, n_trials, n_seeds, n_p_values=8, n_lam_sweep=25):
    alpha_c  = 0.138
    p_values = np.linspace(0.15, 0.85, n_p_values)
    lam_pw   = 0.01   # pairwise limit  (lambda -> 0)
    lam_3b   = 100.0   # 3-body limit    (lambda -> inf)
    p_fixed  = 0.35

    # ── p sweep (pairwise & 3-body limits) ───────────────────────────────────
    th_pw, th_3b, emp_pw, emp_3b = [], [], [], []

    tqdm.write(
        f"\n  {'p':>5}  {'qt':>5}  |  {'pw(th)':>8} {'pw(emp)':>8}"
        f"  |  {'3b(th)':>8} {'3b(emp)':>8}"
    )
    tqdm.write("  " + "-" * 62)

    with tqdm(p_values, desc="  p sweep", unit="p", ncols=90) as pbar:
        for p in pbar:
            _, qt = compute_q_from_budget(p, N, beta)
            pc_pw_th = alpha_c * p  * N
            pc_3b_th = alpha_c * qt * N / 2.0

            pbar.set_postfix(p=f"{p:.2f}", stage="pairwise")
            pc_pw = find_empirical_pc(N, p, beta, lam_pw, n_trials, n_seeds, pbar=pbar)

            pbar.set_postfix(p=f"{p:.2f}", stage="3-body")
            pc_3b = find_empirical_pc(N, p, beta, lam_3b, n_trials, n_seeds, pbar=pbar)

            th_pw.append(pc_pw_th);  th_3b.append(pc_3b_th)
            emp_pw.append(pc_pw);    emp_3b.append(pc_3b)
            tqdm.write(
                f"  {p:5.2f}  {qt:5.2f}  |  {pc_pw_th:8.1f} {pc_pw:8.1f}"
                f"  |  {pc_3b_th:8.1f} {pc_3b:8.1f}"
            )

    th_pw  = np.array(th_pw);  th_3b  = np.array(th_3b)
    emp_pw = np.array(emp_pw, float);  emp_3b = np.array(emp_3b, float)
    rho_pw, _ = spearmanr(th_pw, emp_pw)
    rho_3b, _ = spearmanr(th_3b, emp_3b)
    print(f"  Pairwise rho = {rho_pw:.3f}   3-body rho = {rho_3b:.3f}")

    # ── lambda sweep at fixed p ───────────────────────────────────────────────
    _, qt_fixed = compute_q_from_budget(p_fixed, N, beta)
    lam_sweep   = np.logspace(-2, 2, n_lam_sweep)
    pc_th_sw, pc_emp_sw = [], []

    with tqdm(lam_sweep, desc=f"  lam sweep p={p_fixed}", unit="lam", ncols=90) as pbar:
        for lam in pbar:
            pc_th_sw.append(replica_capacity(p_fixed, N, beta, lam=lam))
            pbar.set_postfix(lam=f"{lam:.3f}")
            pc_emp_sw.append(
                find_empirical_pc(N, p_fixed, beta, lam, n_trials, n_seeds, pbar=pbar)
            )

    # ── Colour-blind-safe palette ─────────────────────────────────────────────
    _C_PW  = "#0077BB"   # blue
    _C_3B  = "#CC3311"   # red
    _C_OPT = "#009988"   # teal

    # ── Dense theory curves ───────────────────────────────────────────────────
    p_dense      = np.linspace(min(p_values) * 0.95, max(p_values) * 1.05, 300)
    qt_dense     = 6.0 * beta - 3.0 * p_dense
    pc_opt_dense = np.array([capacity_on_optimal_line(pp, N, beta) for pp in p_dense])

    lam_opt_fixed = optimal_lambda(p_fixed, beta)
    pc_opt_fixed  = capacity_on_optimal_line(p_fixed, N, beta)

    # ── Figure 1: p sweep ─────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(1, 1, figsize=(9, 5.5), constrained_layout=True)

    # ── Panel (a): p sweep ───────────────────────────────────────────────────
    ax1.plot(p_dense, alpha_c * p_dense * N, "--", color=_C_PW, lw=2.0,
             label=r"Theory: $\alpha_c p N$  ($\lambda\to 0$)")
    ax1.scatter(p_values, emp_pw, c=_C_PW, s=60, zorder=4,
                label=r"Empirical ($\lambda\approx 0$)")

    ax1.plot(p_dense, alpha_c * qt_dense * N / 2, "--", color=_C_3B, lw=2.0,
             label=r"Theory: $\alpha_c\tilde{q}N/2$  ($\lambda\to\infty$)")
    ax1.scatter(p_values, emp_3b, c=_C_3B, s=60, marker="s", zorder=4,
                label=r"Empirical ($\lambda\approx\infty$)")

    ax1.annotate(
        rf"$\rho_{{\rm pw}}={rho_pw:.3f}$   $\rho_{{3\rm b}}={rho_3b:.3f}$",
        xy=(0.97, 0.05), xycoords="axes fraction", ha="right", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#AAAAAA", alpha=0.9),
    )
    ax1.set_xlabel(r"Edge density $p$")
    ax1.set_ylabel(r"Critical capacity $P_c$")
    ax1.set_title(
        rf"(a) Limiting cases — pairwise vs. 3-body  ($N={N},\;\beta={beta}$)")
    ax1.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax1.set_xlim(p_dense[0], p_dense[-1])
    ax1.set_ylim(bottom=0)

    path1 = os.path.join(PLOT_DIR, "limiting_cases_p_sweep.png")
    fig1.savefig(path1)
    plt.close(fig1)
    print(f"  Saved {path1}")

    # ── Figure 2: lambda sweep ────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 5.5), constrained_layout=True)

    ax2.plot(lam_sweep, pc_th_sw,  "-",  color="#222222", lw=2.2,
             label="Replica theory")
    ax2.plot(lam_sweep, pc_emp_sw, "o-", color="#EE7733", lw=1.8,
             ms=6, zorder=4, label="Simulation")

    pc_pw_lim = alpha_c * p_fixed * N
    pc_3b_lim = alpha_c * qt_fixed * N / 2.0

    ax2.axhline(pc_pw_lim, color=_C_PW, ls=":", lw=1.8, alpha=0.85,
                label=rf"Pairwise limit $= {pc_pw_lim:.0f}$")
    ax2.axhline(pc_3b_lim, color=_C_3B, ls=":", lw=1.8, alpha=0.85,
                label=rf"3-body limit $= {pc_3b_lim:.0f}$")
    ax2.axvline(lam_opt_fixed, color=_C_OPT, ls="--", lw=2.0, alpha=0.90,
                label=rf"$\lambda^*={lam_opt_fixed:.2f}$  ($P_c^{{\rm opt}}={pc_opt_fixed:.0f}$)")
    ax2.scatter([lam_opt_fixed], [pc_opt_fixed],
                color=_C_OPT, s=90, zorder=6, clip_on=False)

    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\lambda = b/a$")
    ax2.set_ylabel(r"Critical capacity $P_c$")
    ax2.set_title(
        rf"(b) Capacity vs. $\lambda$  ($p={p_fixed},\;N={N},\;\beta={beta}$)")
    ax2.legend(loc="center right", framealpha=0.9)
    ax2.set_ylim(bottom=0)
    ax2.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"${y:g}$"))

    path2 = os.path.join(PLOT_DIR, "limiting_cases_lambda_sweep.png")
    fig2.savefig(path2)
    plt.close(fig2)
    print(f"  Saved {path2}")

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_all(quick=False, profile=None):
    if profile is None:
        profile = "quick" if quick else "full"

    if profile == "quick":
        N, n_p, n_lam, n_trials, n_seeds = 70, 6, 6, 2, 2
        n_lim_p, n_lim_lam = 8, 30
    elif profile == "large":
        N, n_p, n_lam, n_trials, n_seeds = 100, 13, 13, 12, 12
        n_lim_p, n_lim_lam = 12, 60
    elif profile == "N1000":
        # N=1000: sparse topology keeps memory manageable;
        # fewer grid points keep total wall-time tractable.
        N, n_p, n_lam, n_trials, n_seeds = 1000, 8, 10, 10, 3
        n_lim_p, n_lim_lam = 6, 35
    else:
        N, n_p, n_lam, n_trials, n_seeds = 100, 13, 13, 8, 5
        n_lim_p, n_lim_lam = 12, 50
    beta = 0.5

    print(f"Backend: torch  device: {_DEVICE}  "
          f"N={N}  [{profile}]")

    print("\n── A: Capacity heatmap in (p, lambda) ─────────────────────")
    heatmap_p_lambda(
        N=N, beta=beta, n_p=n_p, n_lam=n_lam,
        n_trials=n_trials, n_seeds=n_seeds,
    )

    print("\n── B: Limiting cases ───────────────────────────────────────")
    limiting_cases(
        N=N, beta=beta, n_trials=n_trials, n_seeds=n_seeds,
        n_p_values=n_lim_p, n_lam_sweep=n_lim_lam,
    )

if __name__ == "__main__":
    run_all()
