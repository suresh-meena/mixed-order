from __future__ import annotations

import sys
import pathlib
_file = pathlib.Path(__file__).resolve()
_repo_root = None
for _ancestor in _file.parents:
    if _ancestor.name == "experiments":
        _repo_root = _ancestor.parent
        break
if _repo_root is None:
    _repo_root = _file.parents[1] if len(_file.parents) >= 2 else _file.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pathlib import Path

from experiments.common import apply_pub_style, get_color_palette, load_npz, plt, save_fig

RESULT_DIR = Path(__file__).resolve().parent


def main() -> None:
    apply_pub_style()
    cp = get_color_palette()
    data = load_npz(RESULT_DIR / "comparison_pairwise_mixed_results.npz")

    alpha = data["alpha_vals"]
    classical = data["classical_success"]
    mixed_opt = data["mixed_opt_success"]
    mixed_opt_ref = data["mixed_opt_ref_success"]
    lam_opt = float(data["lam_opt"][0])
    p_ref = float(data["p_ref"][0])
    lam_opt_ref = float(data["lam_opt_ref"][0])
    N = int(data["N"][0])
    p = float(data["p"][0])

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(alpha, classical, "o-", color=cp["uncentered"], lw=2.2, label="classical pairwise")
    ax.plot(alpha, mixed_opt, "^-", color=cp["centered"], lw=2.2, label=rf"mixed order optimal ($\lambda^*={lam_opt:.1f}$)")
    ax.plot(alpha, mixed_opt_ref, "d-", color=cp["extra"], lw=2.2, label=rf"mixed order optimal at $p={p_ref:.1f}$ ($\lambda^*={lam_opt_ref:.1f}$)")
    ax.axhline(0.99, color=cp["theory"], ls="--", lw=1.6, label="success threshold")
    ax.set_xlabel(r"Load $\alpha=P/N$")
    ax.set_ylabel("Retrieval success")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(rf"Storage capacity vs load  ($N={N},\;p={p:.2f}$)")
    ax.legend(frameon=False)
    save_fig(fig, RESULT_DIR / "comparison_pairwise_mixed_capacity.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
