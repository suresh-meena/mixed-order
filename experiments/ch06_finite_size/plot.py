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
    data = load_npz(RESULT_DIR / "ch06_results.npz")

    N = data["n_values"].astype(int)
    pc_over_n = data["pc_over_n"]
    lam_star = data["lam_star_struct"]
    alpha_star = data["alpha_star_learn"]

    fig_pc, ax_pc = plt.subplots(figsize=(8.6, 5.0))
    ax_pc.plot(N, pc_over_n, "o-", color=cp["mixed"], label=r"$P_c/N$")
    ax_pc.set_xlabel("N")
    ax_pc.set_ylabel(r"$P_c/N$")
    ax_pc.set_title("Chapter 6: finite-size robustness of capacity")
    ax_pc.legend(frameon=False)
    save_fig(fig_pc, RESULT_DIR / "ch06_pc_over_n.png")
    plt.close(fig_pc)

    fig_params, ax_params = plt.subplots(figsize=(8.6, 5.0))
    ax_params.plot(N, lam_star, "o-", color=cp["centered"], label=r"$\lambda^*_{struct}$")
    ax_params.plot(N, alpha_star, "s--", color=cp["extra"], label=r"$\alpha^*_{learn}$")
    ax_params.set_xlabel("N")
    ax_params.set_ylabel("critical parameter")
    ax_params.set_title("Finite-size critical parameters")
    ax_params.legend(frameon=False)
    save_fig(fig_params, RESULT_DIR / "ch06_critical_params.png")
    plt.close(fig_params)


if __name__ == "__main__":
    main()
