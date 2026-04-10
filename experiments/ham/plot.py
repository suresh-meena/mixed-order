from __future__ import annotations

import os
import numpy as np

from experiments.plot_helpers import apply_pub_style, get_color_palette, save_fig, plt

RESULT_DIR = os.path.dirname(__file__)


def _load_results(path: str) -> dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}


def _task_dict(prefix: str, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key[len(prefix) :]: value for key, value in data.items() if key.startswith(prefix)}


def main() -> None:
    apply_pub_style()
    cp = get_color_palette()
    data = _load_results(os.path.join(RESULT_DIR, "ham_suite_results.npz"))

    task3 = _task_dict("task3_", data)
    task1 = _task_dict("task1_", data)
    task2 = _task_dict("task2_", data)
    task4 = _task_dict("task4_", data)

    # Task 3: antipodal null-space (bar plot)
    x = np.arange(2)
    width = 0.23
    labels = ["A", "B"]
    fig_t3, ax_t3 = plt.subplots(figsize=(6.5, 4.5))
    ax_t3.bar(x - width, task3["pairwise"], width, label="pairwise", color=cp["uncentered"])
    ax_t3.bar(x, task3["cubic"], width, label="cubic", color=cp["centered"])
    ax_t3.bar(x + width, task3["mixed"], width, label="mixed", color=cp["mixed"])
    ax_t3.set_xticks(x)
    ax_t3.set_xticklabels(labels)
    ax_t3.set_title("Task 3: antipodal null-space")
    ax_t3.set_ylabel("score")
    ax_t3.legend(frameon=False)
    ax_t3.text(0.02, 0.02, "pairwise ties\ncubic flips sign", transform=ax_t3.transAxes, fontsize=10, va="bottom")
    save_path_t3 = os.path.join(RESULT_DIR, "ham_task3.png")
    fig_t3.tight_layout()
    fig_t3.savefig(save_path_t3)
    plt.close(fig_t3)
    print(f"saved report: {save_path_t3}")

    # Task 1: parity retrieval (line plot)
    fig_t1, ax_t1 = plt.subplots(figsize=(6.5, 4.5))
    colors = {"pairwise": cp["uncentered"], "cubic": cp["centered"], "mixed": cp["mixed"]}
    for mode in ("pairwise", "cubic", "mixed"):
        ax_t1.plot(task1["noise_levels"], task1[f"accuracy_{mode}"], "o-", color=colors[mode], label=mode)
    ax_t1.set_title("Task 1: parity retrieval")
    ax_t1.set_xlabel("bit-flip noise")
    ax_t1.set_ylabel("accuracy")
    ax_t1.set_ylim(0.0, 1.02)
    ax_t1.legend(frameon=False)
    save_path_t1 = os.path.join(RESULT_DIR, "ham_task1.png")
    fig_t1.tight_layout()
    fig_t1.savefig(save_path_t1)
    plt.close(fig_t1)
    print(f"saved report: {save_path_t1}")

    # Task 4: spurious attractor census
    fig_t4, ax_t4 = plt.subplots(figsize=(6.5, 4.5))
    ax_t4.plot(task4["k_grid"], task4["pairwise_spurious"], "o-", color=cp["uncentered"], label="pairwise")
    ax_t4.plot(task4["k_grid"], task4["mixed_spurious"], "o-", color=cp["mixed"], label="mixed")
    ax_t4.set_title("Task 4: spurious attractor census")
    ax_t4.set_xlabel("stored patterns K")
    ax_t4.set_ylabel("unique non-stored attractors")
    ax_t4.legend(frameon=False)
    save_path_t4 = os.path.join(RESULT_DIR, "ham_task4.png")
    fig_t4.tight_layout()
    fig_t4.savefig(save_path_t4)
    plt.close(fig_t4)
    print(f"saved report: {save_path_t4}")

    # Task 2: load curve at fixed budget
    fig_t2, ax_t2 = plt.subplots(figsize=(6.5, 4.5))
    ax_t2.plot(task2["alpha_grid"], task2["iid_pairwise"], "o--", color=cp["uncentered"], label="iid pairwise")
    ax_t2.plot(task2["alpha_grid"], task2["iid_mixed"], "o-", color=cp["mixed"], label="iid mixed")
    ax_t2.plot(task2["alpha_grid"], task2["structured_pairwise"], "s--", color=cp["extra"], label="structured pairwise")
    ax_t2.plot(task2["alpha_grid"], task2["structured_mixed"], "s-", color=cp["centered"], label="structured mixed")
    ax_t2.set_title("Task 2: load curve at fixed budget")
    ax_t2.set_xlabel(r"$\alpha = P / N^2$")
    ax_t2.set_ylabel("classification accuracy")
    ax_t2.set_ylim(0.0, 1.02)
    ax_t2.legend(frameon=False, fontsize=9)
    save_path_t2 = os.path.join(RESULT_DIR, "ham_task2.png")
    fig_t2.tight_layout()
    fig_t2.savefig(save_path_t2)
    plt.close(fig_t2)
    print(f"saved report: {save_path_t2}")


if __name__ == "__main__":
    main()