from __future__ import annotations

import os
import numpy as np

from experiments.plot_helpers import apply_pub_style, get_color_palette, save_fig, plt

RESULT_DIR = os.path.dirname(__file__)


def main() -> None:
    apply_pub_style()
    data = np.load(os.path.join(RESULT_DIR, "learning_boundary_results.npz"), allow_pickle=True)
    N = int(data["N"]) if "N" in data.files else 1000
    lambda_vals = data["lambda_vals"]
    alpha_crossings = data["alpha_crossings"]
    alpha_cis = data["alpha_cis"]
    c_fitted = float(data["c_fitted"])
    g2 = float(data["g2"])

    def scaling_law(lam: np.ndarray, c: float) -> np.ndarray:
        return c * (1 + lam / 2.0) / np.sqrt(g2)

    theory_overlay = scaling_law(lambda_vals, c_fitted)
    cp = get_color_palette()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(lambda_vals, alpha_crossings, yerr=alpha_cis, fmt="o", color=cp["mixed"], capsize=5, label="Measured Crossing")
    ax.plot(lambda_vals, theory_overlay, color=cp["theory"], lw=2, ls="--", label=rf"Scaling: ${c_fitted:.2f} \cdot (1+\lambda/2)/\sqrt{{g_2}}$")
    ax.set_xlabel(r"Coupling ratio $\lambda = b/a$")
    ax.set_ylabel(r"Learning boundary $\alpha^*$")
    ax.set_title(rf"Storage-to-Learning Transition ($N={N}, g_2={g2:.2f}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(RESULT_DIR, "learning_boundary.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved plot to {path}")


if __name__ == "__main__":
    main()