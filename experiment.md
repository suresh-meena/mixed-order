# Experiment Plan

Goal: a minimal, self-contained set of numerical experiments that proves the thesis claims chapter by chapter. Every experiment must support a specific statement in the text. No exploratory runs, no decorative diagnostics, no duplicate evidence.

## Common conventions

- Use the same code paths for pattern generation, learning, retrieval, centering, and metrics across all chapters.
- Default retrieval metric: overlap \(m = (1/N)\,\xi \cdot \sigma_{\mathrm{final}}\).
- Success threshold: \(m > 0.9\) unless a chapter needs a stricter threshold.
- Define \(P_c\) as the largest load \(P\) whose success rate is at least 90% across seeds.
- Use \(N \in \{200, 500, 1000\}\) for finite-size checks and \(N=500\) for fast sweeps unless a chapter needs a different scale.
- Compare against the traditional pairwise Hopfield model only when it isolates the effect being claimed. Do not add a Hopfield baseline if it does not change the conclusion.
- For the mixed-order model in Chapter 2, enforce the exact budget
  \[
  q = \frac{6 - 3p}{N}
  \]
  so the total active parameter count stays at \(M = \binom{N}{2}\).

## Chapter 1: Introduction (`chapters/01_introduction.tex`)

### 1.1 Classical Hopfield threshold

What it tests:
- The classical Hopfield limit \(\alpha_c \approx 0.138\) for i.i.d. Rademacher patterns.

Setup:
- Sweep \(\alpha = P/N\) for \(N \in \{200, 500, 1000\}\).
- Use several seeds per point.
- Run the same retrieval dynamics used everywhere else in the thesis.

Measure:
- Retrieval success rate.
- Final overlap \(m\).
- \(P_c\) as a function of \(N\).

Plots:
- Success rate vs \(\alpha\) for each \(N\).
- Finite-size plot of \(P_c/N\) vs \(N\) with the asymptotic reference line at \(0.138\).

Comparison note:
- This is the baseline itself, so no extra model comparison is needed here.

### 1.2 Update-rule sanity check

What it tests:
- Asynchronous and synchronous updates agree in the subcritical regime and do not change the baseline conclusion.

Setup:
- Pick one subcritical \(\alpha\) and one near-critical \(\alpha\) from Experiment 1.1.
- Run both `async` and `sync` updates on the same patterns and seeds.

Measure:
- Success rate.
- Mean final overlap.
- One representative energy trace for `async`.

Plots:
- Overlaid success-rate curves for `async` and `sync`.
- One energy-vs-iteration trace showing monotone descent for `async`.

Comparison note:
- Do not add a separate traditional Hopfield control here. The point is only to check update dynamics on the same baseline model.

## Chapter 1b: Hopfield Networks (`chapters/01b_hopfield_networks.tex`)

- No standalone experiment.
- Reuse Experiments 1.1 and 1.2 as the numerical support for dynamics, convergence, and the retrieval metric.

## Chapter 2: Baseline Architecture (`chapters/02_baseline_architecture.tex`)

### 2.1 Mixed-order capacity under the exact budget

What it tests:
- The mixed pairwise + 3-body model gains capacity under the fixed budget \(M = \binom{N}{2}\).
- The optimum is not the formal \(p \to 0\) endpoint because the graph may lose connectivity first.

Setup:
- Fix \(N\).
- Sweep \(p \in [0,1]\).
- Enforce \(q = (6 - 3p)/N\).
- Sweep \(\lambda = b/a\) over a small grid.
- Keep the same retrieval threshold and seeds across all points.
- Include the traditional pairwise Hopfield model only as a reference curve on the same axes, if it helps show the capacity gain.

Measure:
- \(P_c(p,\lambda)\).
- Success rate.
- Final overlap.
- Empirical optimal \(p^*(\lambda)\).

Plots:
- \(P_c\) vs \(p\) for several representative \(\lambda\) values.
- Heatmap of \(P_c(p,\lambda)\).
- Vertical marker for the low-\(p\) connectivity cutoff near \(p \sim \log N / N\).
- Optional reference curve: classical pairwise Hopfield \(P_c\) at the same \(N\), shown only if it helps make the gain obvious.

Required conclusion:
- The best point must lie in the connected regime.
- The numeric optimum should be interior rather than at \(p \to 0\).

### 2.2 Signal-noise decomposition

What it tests:
- The chapter’s signal and crosstalk formulas match the measured local fields.

Setup:
- Use a few representative points from Experiment 2.1.
- Measure fields near retrieval, ideally at \(m \approx 1\).
- Average over patterns, seeds, and neurons.
- No separate Hopfield comparison is needed unless a control run is required to show the mixed-order variance reduction.

Measure:
- Empirical signal \(S_{\mathrm{emp}}\).
- Empirical crosstalk variance \(\tau_{\mathrm{emp}}^2\).

Plots:
- Theory vs empirical overlays for \(S\) and \(\tau^2\) as functions of \(P\).
- One residual plot for crosstalk error.

### 2.3 Gram-matrix diagnostic

What it tests:
- The Gram interpretation of signal and crosstalk in feature space.

Setup:
- Reuse the same runs as Experiment 2.1.
- Build the mixed feature map \(\Phi\) and Gram matrix \(K = \Phi \Phi^\top\).

Measure:
- Diagonal mass.
- Off-diagonal variance.
- Their ratio as \(P\) grows.

Plots:
- Diagonal vs off-diagonal mass across \(P\).
- Or a two-panel plot with diagonal and off-diagonal summaries.

Comparison note:
- Compare only against the pairwise Hopfield Gram picture if the chapter needs a direct classical reference; otherwise keep this as an internal mixed-order diagnostic.

## Chapter 3: Structured Data (`chapters/03_structured_data.tex`)

### 3.1 Gaussian-sign covariance strength

What it tests:
- The Gaussian-sign teacher generates the covariance correction \(g_2\) used in the chapter.

Setup:
- Generate patterns with \(D = \gamma N\).
- Use \(\gamma \in \{0.5, 1\}\).
- Compute the empirical covariance matrix \(C\).

Measure:
- \(g_2 = (1/N)\sum_{i \neq j} C_{ij}^2\).

Plots:
- \(g_2\) vs \(\gamma\).
- Optional scaling check against \(1/D\).

Comparison note:
- No traditional Hopfield comparison is needed here; the experiment is only about the structured teacher itself.

### 3.2 Centering and optimal-\(\lambda\) shift

What it tests:
- Pairwise centering removes the deterministic drift.
- The optimal coupling ratio shifts by \(\Delta \lambda^* = \tilde q\, g_2\).

Setup:
- Run the same structured-data sweep with and without pairwise centering.
- Sweep \(\lambda\) at fixed \(p\).
- Choose a drift scale large enough to shift the optimum but not so large that retrieval breaks.
- If a classical pairwise Hopfield control helps, use it only as the uncentered pairwise reference.

Measure:
- \(P_c(\lambda)\) before and after centering.
- Empirical optimal \(\lambda^*\).
- \(\Delta \lambda^*\).

Plots:
- \(P_c\) vs \(\lambda\) for centered and uncentered weights on the same axes.
- Scatter plot of measured \(\Delta \lambda^*\) vs \(\tilde q g_2\) with a fitted line.

Comparison note:
- The meaningful control here is uncentered pairwise storage; do not add a separate Hopfield curve unless it clarifies the centering effect.

## Chapter 4: Storage Versus Learning (`chapters/04_learning_phase.tex`)

### 4.1 Storage-learning boundary

What it tests:
- The transition from associative storage to feature learning.
- The scaling \(\alpha^*_{\mathrm{learn}}(\lambda) \propto (1+\lambda/2)/\sqrt{g_2}\).

Setup:
- Choose a few representative \((p,\lambda)\) pairs from Chapter 3.
- Sweep \(\alpha = P/N\) upward until retrieval fails.
- Compare against the pairwise-only baseline at the same \((p,\alpha)\) only when it helps show that the cubic channel delays the storage-learning crossover.

Measure:
- Retrieval success.
- Final overlap.
- Deterministic drift strength.
- An alignment metric such as \(\mathrm{align}(J,C)\).

Plots:
- Retrieval success vs \(\alpha\).
- Drift vs \(\alpha\).
- Extracted \(\alpha^*_{\mathrm{learn}}\) vs \(\lambda\) with the predicted scaling curve.

Comparison note:
- The pairwise Hopfield control is optional here and should only be included if it makes the storage-learning separation clearer.

### 4.2 Two-dimensional phase map

What it tests:
- The storage, learning, and failure regions as a function of load and cubic strength.

Setup:
- Sweep a grid in \((\alpha,\lambda)\) for the same teacher family.
- Classify each point by retrieval outcome.

Measure:
- Retrieval rate.
- Final-state overlap statistics.

Plots:
- Phase diagram in \((\alpha,\lambda)\) with the regions labeled.
- Boundary curve overlaid on the same plot.

Comparison note:
- No separate traditional Hopfield plot is needed unless a one-dimensional pairwise slice is useful as a reference line.

## Chapter 5: HAM Formalisation (`chapters/05_ham_formalisation.tex`)

### 5.1 Pairwise vs mixed-order HAM on a higher-order task

What it tests:
- The mixed-order HAM solves a task that the pairwise HAM cannot solve.
- The higher-order score family is not just a reformulation; it adds representational power.

Task design:
- Use a parity-style retrieval task built from three latent bits.
- Let the target depend on a true 3-way interaction, for example
  \[
  y = u_1 u_2 u_3
  \]
  in \(\{-1,+1\}\) coding, or an equivalent XOR construction.
- Embed the task so that pairwise statistics alone do not determine the target, but the cubic feature does.

Setup:
- Compare a pairwise HAM against a mixed-order HAM at the same budget \(M = \binom{N}{2}\).
- Use the same cues, noise corruption, and evaluation set for both models.
- Sweep the corruption level or attention temperature \(\beta\).
- This is the main place where the traditional pairwise model must be shown, because the claim is that only the mixed-order HAM solves the higher-order task.

Measure:
- Retrieval accuracy.
- Final overlap.
- Basin size or basin radius.

Plots:
- Accuracy vs cue corruption for pairwise vs mixed-order HAM.
- Basin-size or basin-radius comparison.
- One representative example showing the pairwise model failing where the mixed-order model succeeds.

Comparison note:
- Keep the pairwise HAM comparison here because it is the direct evidence for the added value of mixed-order features.

Required conclusion:
- The pairwise HAM should remain at chance or fail sharply on the task.
- The mixed-order HAM should recover the correct target significantly above the pairwise baseline.

## Chapter 6: Conclusion (`chapters/06_conclusion.tex`)

### 6.1 Finite-size robustness

What it tests:
- The main trends survive finite-size changes.

Setup:
- Rerun the top configurations from Chapters 2 to 5 at \(N \in \{200, 500, 1000\}\).

Measure:
- \(P_c\).
- \(\lambda^*_{\mathrm{struct}}\).
- \(\alpha^*_{\mathrm{learn}}\).

Plots:
- Finite-size scaling plots for \(P_c\) and the critical parameters versus \(N\).

Comparison note:
- Use the same model family as in the earlier chapters; do not add separate traditional Hopfield baselines unless you are checking the finite-size behavior of the baseline itself.

## Minimal implementation order

1. Chapter 2 mixed-order capacity sweep under the exact budget.
2. Chapter 3 covariance strength and centering shift.
3. Chapter 4 storage-learning boundary and phase map.
4. Chapter 1 baseline Hopfield threshold and update-rule check.
5. Chapter 5 pairwise vs mixed-order HAM on the higher-order task.
6. Chapter 6 finite-size validation.
