# Thesis Experiment Implementation Plan

This is the implementation plan for the `mixed-order` experiment layer. It is intentionally narrow. The goal is to produce the smallest experiment set that fully supports the claims in `ms-writeup` and nothing more.

The final thesis experiment package should consist of exactly four main figures:

1. baseline capacity surface
2. structured-data correction and optimum shift
3. measured storage-learning boundary
4. HAM synthetic higher-order task

The current codebase already contains partial support for the first three. The HAM experiment must be replaced.

## 1. Scope And Claim Mapping

Use the thesis claims as the filter.

From `ms-writeup`, the empirical claims worth testing are:

- Chapter 2:
  the mixed-order architecture has a nontrivial `P_c(p, lambda)` under an `O(N^2)` budget, with an optimum ridge `lambda^*(p) = \tilde q / p`
- Chapter 3:
  structured data creates pairwise drift, centering removes that drift, and the covariance strength shifts the best `lambda`
- Chapter 4:
  the learning boundary scales roughly like `(1 + lambda/2) / sqrt(g2)`
- Chapter 5:
  the sparse mixed-order score family matters on a task with true higher-order structure

Anything that does not support one of those four statements should not be a main-text figure.

## 2. Current Experiment Status

Current scripts:

- keep as base:
  - `/home/suresh/SHN/mixed-order/experiments/capacity_heatmap/capacity_heatmap.py`
  - `/home/suresh/SHN/mixed-order/experiments/structured/structured_capacity.py`
  - `/home/suresh/SHN/mixed-order/experiments/learning/phase_diagram.py`
- move out of main path:
  - `/home/suresh/SHN/mixed-order/experiments/limiting_cases/limiting_cases.py`
- replace:
  - `/home/suresh/SHN/mixed-order/experiments/ham/softmax_retrieval.py`

Current shared code to reuse:

- `/home/suresh/SHN/mixed-order/src/mixed_order/theory.py`
- `/home/suresh/SHN/mixed-order/src/mixed_order/metrics.py`
- `/home/suresh/SHN/mixed-order/src/mixed_order/data/structured.py`
- `/home/suresh/SHN/mixed-order/src/mixed_order/storage.py`
- `/home/suresh/SHN/mixed-order/src/mixed_order/topology.py`
- `/home/suresh/SHN/mixed-order/src/mixed_order/plotting/style.py`

Main weakness in the current experiment layer:

- `P_c` is inferred from mean overlap only
- confidence intervals are not handled cleanly
- the structured experiment is too narrow
- the learning script produces an explanatory plot, not the actual scaling figure claimed in the thesis
- the HAM script is a random-memory recovery demo and does not isolate higher-order necessity

## 3. Global Experimental Rules

These rules must be applied consistently across all thesis experiments.

### 3.1 Retrieval Metric

Use the same success definition everywhere:

- query corruption:
  flip `10%` of bits for capacity experiments unless noise is explicitly swept
- retrieval success:
  final overlap `m = (1/N) x·xi >= 0.95`
- critical capacity:
  largest `P` such that success rate is at least `0.9`

Do not use mean overlap alone as the definition of `P_c`.

### 3.2 Randomness Control

Use one consistent seed protocol:

- one topology per `(N, p, seed)`
- one pattern bank per `(data setting, seed)`
- reuse the same seed list across all `lambda` values within the same sweep
- do not regenerate topology independently inside each `lambda` point if comparison across `lambda` is the point

### 3.3 Output Artifacts

Every thesis script should save two things:

- one figure file: `.png` and optionally `.pdf`
- one raw result file: `.npz` or `.json`

The raw file should include:

- config values
- seed list
- x-axis grid
- means
- confidence intervals
- any fitted coefficient used in a theory overlay

### 3.4 Plot Policy

All plots should be information-dense and minimal:

- no redundant sanity figures in the main text
- no more than one legend per figure
- same axis scale across directly compared panels
- theory in black or charcoal
- centered / corrected in blue or teal
- uncentered / drift in red
- mixed-order highlight in green
- confidence bands on empirical line plots
- no standalone scatter plots unless uncertainty itself is the message

## 4. Shared Refactor Before Any Figure Work

This is the first implementation block. Do this before editing the individual scripts.

### 4.1 `metrics.py`

File to modify:

- `/home/suresh/SHN/mixed-order/src/mixed_order/metrics.py`

Add the following helpers:

- `compute_overlap(finals, targets) -> Tensor`
- `compute_success(overlaps, threshold=0.95) -> Tensor[bool]`
- `aggregate_success(success, dim=...) -> rate`
- `bootstrap_ci(values, confidence=0.95, n_boot=1000)`
- `find_empirical_pc_by_success(...)`

`find_empirical_pc_by_success(...)` should replace the current boundary logic that walks mean overlap.

Required behavior:

- evaluate success rate for each candidate `P`
- find the largest `P` with success rate `>= success_threshold`
- optionally return the full sweep arrays for plotting and diagnostics
- support structured data by passing through `F`, `C`, `centered`

Keep the current batch execution pattern. Do not redesign the full engine here.

### 4.2 Result Containers

Add small result dictionaries in the scripts rather than creating a large framework.

Each script should save:

- config dictionary
- theory curve or surface
- empirical curve or surface
- confidence intervals if present

Do not add a heavy experiment manager. The codebase should stay flat and simple.

### 4.3 Naming

Keep the current files where practical. Rename only when the script meaning changes materially.

Recommended final script names:

- `/home/suresh/SHN/mixed-order/experiments/capacity_heatmap/capacity_heatmap.py`
- `/home/suresh/SHN/mixed-order/experiments/structured/structured_capacity.py`
- `/home/suresh/SHN/mixed-order/experiments/learning/learning_boundary.py`
- `/home/suresh/SHN/mixed-order/experiments/ham/xor3_hybrid_task.py`

This means:

- `phase_diagram.py` should be replaced by `learning_boundary.py`
- `softmax_retrieval.py` should be replaced by `xor3_hybrid_task.py`

## 5. Figure 1: Baseline Capacity Surface

### 5.1 Claim

This figure supports Chapter 2:

- theory predicts `P_c(p, lambda)`
- theory predicts the optimum ridge `lambda^*(p) = \tilde q / p`

### 5.2 File To Edit

- `/home/suresh/SHN/mixed-order/experiments/capacity_heatmap/capacity_heatmap.py`

### 5.3 What To Keep

Keep the existing structure:

- analytical surface from `theory.py`
- empirical grid sweep
- two-panel heatmap
- theory ridge overlay

### 5.4 What To Change

Change the empirical capacity estimator:

- use `find_empirical_pc_by_success(...)`
- not the current mean-overlap threshold logic

Add raw output saving:

- `heatmap_p_lambda_results.npz`

Tighten the plotting:

- same color scale in both panels
- one shared colorbar if possible
- annotate the empirical panel with:
  - mean absolute percentage error
  - number of seeds
  - success threshold

### 5.5 Parameter Grid

Use:

- `N = 256`
- `beta = 0.5`
- `p_vals = np.linspace(0.15, 0.75, 8)`
- `lambda_vals = np.logspace(np.log10(0.2), np.log10(6.0), 10)`
- `n_trials = 16`
- `n_seeds = 8`

### 5.6 Final Plot

Single figure with two panels:

- panel A:
  theory heatmap
- panel B:
  empirical heatmap

This is the only Chapter 2 experiment needed in the main text.

### 5.7 What To Remove From Main Text

- `/home/suresh/SHN/mixed-order/experiments/limiting_cases/limiting_cases.py`

That script can remain in the repo, but it should not be a thesis figure once the heatmap exists.

## 6. Figure 2: Structured Correction And Optimum Shift

### 6.1 Claim

This figure supports Chapter 3:

- uncentered pairwise storage drifts on structured data
- centering removes the drift
- stronger covariance shifts the optimal `lambda` upward

### 6.2 File To Edit

- `/home/suresh/SHN/mixed-order/experiments/structured/structured_capacity.py`

### 6.3 Core Design

Turn this into a two-panel figure.

Panel A:

- drift amplitude vs load at `lambda = 0`
- compare:
  - uncentered pairwise
  - centered pairwise

Panel B:

- centered `P_c(lambda)` curves for three covariance strengths
- one extra uncentered curve only for the strongest-structure condition
- vertical lines at theoretical `lambda^*_{struct}`

### 6.4 Data Settings

Generate Gaussian-sign data with three latent dimensions:

- `D = 32`
- `D = 64`
- `D = 128`

This gives low / medium / high structure in a simple way. If the measured `g2` ordering does not separate enough, choose a wider spread.

### 6.5 Shared Quantities

For each `D`:

- generate `F`
- compute `C`
- estimate `g2`
- compute `lambda^*_{struct}(p, g2)`

Save these values into the raw result file.

### 6.6 Drift Measurement

You already have drift logic in:

- `/home/suresh/SHN/mixed-order/experiments/learning/phase_diagram.py`

Do not duplicate that code ad hoc. Extract or locally factor a helper:

- `measure_pairwise_drift_vs_load(...)`

For this figure, use:

- `lambda = 0`
- one representative `D`, preferably the strongest-structure setting
- plot both centered and uncentered drift

### 6.7 Capacity Sweep

For each `D`:

- sweep `lambda` on log scale
- estimate centered `P_c`
- compute seed-wise confidence bands

Add one uncentered curve only for the lowest `D` or strongest `g2` condition, whichever shows the failure most clearly.

### 6.8 Parameters

Use:

- `N = 256`
- `p = 0.35`
- `beta = 0.5`
- `lambda_vals = np.logspace(np.log10(0.2), np.log10(8.0), 11)`
- `n_trials = 16`
- `n_seeds = 6`

### 6.9 Final Plot

One figure with two panels:

- panel A:
  drift vs load, centered vs uncentered
- panel B:
  `P_c` vs `lambda` for three `D` settings with theory vertical markers

This single figure covers all of Chapter 3.

## 7. Figure 3: Learning Boundary

### 7.1 Claim

This figure supports Chapter 4:

- the boundary scales approximately as `(1 + lambda/2) / sqrt(g2)`

The current `phase_diagram.py` does not test that claim directly. It only shows signal and drift traces for a few `lambda`.

### 7.2 File To Replace

Replace:

- `/home/suresh/SHN/mixed-order/experiments/learning/phase_diagram.py`

with:

- `/home/suresh/SHN/mixed-order/experiments/learning/learning_boundary.py`

### 7.3 Reuse From Existing Code

Reuse the current signal and drift measurement logic:

- pairwise signal
- cubic signal
- total signal
- drift amplitude

Do not keep the current three-panel `lambda in {0,2,10}` figure as the main result.

### 7.4 Boundary Definition

For each `lambda`:

1. sweep `P`
2. compute mean total signal and mean drift amplitude
3. define `alpha^*_{cross}` as the first `alpha = P/N` where
   `total_signal <= drift`

If no crossing occurs in the scanned range, record right-censoring and extend `P_max` or drop that point.

### 7.5 Theory Overlay

The theory gives scaling, not a fixed prefactor.

So the overlay should be:

- `c * (1 + lambda/2) / sqrt(g2)`

where `c` is fitted once using either:

- least squares over all empirical points
- or one anchor point

Least squares is cleaner.

### 7.6 Parameters

Use one structured setting only:

- `N = 256`
- `p = 0.35`
- `beta = 0.5`
- `D = 64`
- `lambda_vals = np.linspace(0.0, 6.0, 9)`
- `n_trials = 12`
- `n_seeds = 6`

Choose `P_max` so the crossing exists for all or almost all `lambda` values.

### 7.7 Optional Inset

If needed, add one inset with the actual signal and drift traces for:

- `lambda = 0`
- `lambda = 4` or another mid/high point

Do not make this a separate full figure.

### 7.8 Final Plot

Single panel:

- x-axis:
  `lambda`
- y-axis:
  measured `alpha^*_{cross}`
- black dashed line:
  fitted scaling law
- empirical points with confidence intervals

This is the Chapter 4 result.

## 8. Figure 4: HAM Higher-Order Necessity

### 8.1 Claim

This figure supports Chapter 5 in the only way that matters empirically:

- the mixed score family helps on a task with true higher-order structure

It should not try to prove a broad HAM theorem.

### 8.2 File To Replace

Replace:

- `/home/suresh/SHN/mixed-order/experiments/ham/softmax_retrieval.py`

with:

- `/home/suresh/SHN/mixed-order/experiments/ham/xor3_hybrid_task.py`

### 8.3 Why The Current Script Is Not Good Enough

The current HAM script stores random patterns and performs softmax retrieval. That only shows the layer runs. It does not show why higher-order structure is necessary.

### 8.4 Task Design

Use a hybrid prototype-plus-parity classification or retrieval task.

State layout:

- prototype block:
  ordinary binary prototype bits, class-specific
- parity block:
  disjoint triples of bits

For each class:

- assign a prototype vector on the prototype block
- assign a parity signature `y_c in {-1, +1}^T` over `T` disjoint triples

For each parity triple:

- sample uniformly from the four spin assignments with product equal to the class parity bit

This guarantees:

- zero mean per parity bit
- zero pairwise correlation inside the parity block
- nonzero third-order moment equal to the class parity label

That is the key property. Pairwise statistics on the parity block are uninformative by construction.

### 8.5 Models To Compare

Compare three score families:

- pairwise-only HAM
- cubic-only HAM
- mixed-order HAM

Define the scores directly from templates. Do not force this experiment through Hebbian storage.

Recommended score decomposition:

- pairwise score:
  prototype similarity plus optional pairwise parity terms
- cubic score:
  sum of triple products on the parity triples
- mixed score:
  weighted sum of both

### 8.6 Query Generation

For a query from class `c`:

- sample a parity-consistent realization on the parity block
- copy the prototype block for class `c`
- corrupt the full state by flipping a fraction of bits

Sweep bit-flip noise over:

- `noise in [0.0, 0.3]`

### 8.7 Expected Outcome

Expected ranking:

- pairwise-only:
  uses prototype bits but cannot decode the parity block
- cubic-only:
  decodes parity but is weaker on pure prototype robustness
- mixed:
  uses both and should dominate across the useful noise range

If the pairwise-only baseline is too strong because the prototype block dominates, reduce prototype size. If cubic-only is too strong everywhere, increase prototype ambiguity or reduce cubic block count. Tune the task until each model expresses its intended inductive bias.

### 8.8 Parameters

Recommended initial setting:

- number of classes: `K = 8` or `16`
- prototype block size: `N_proto = 24` to `48`
- parity triples: `T = 8` to `16`
- total dimension:
  `N = N_proto + 3*T`
- random task instances:
  `20` to `50`
- noise grid:
  `7` to `9` points in `[0, 0.3]`

### 8.9 Final Plot

One line plot:

- x-axis:
  bit-flip probability
- y-axis:
  accuracy
- curves:
  pairwise-only, cubic-only, mixed
- confidence bands:
  across task instances

That is the only HAM figure needed.

## 9. Implementation Order

Implement in this order.

### Step 1

Refactor `/home/suresh/SHN/mixed-order/src/mixed_order/metrics.py`:

- add success-rate based capacity helpers
- add confidence interval helpers
- keep existing batch retrieval path

### Step 2

Edit `/home/suresh/SHN/mixed-order/experiments/capacity_heatmap/capacity_heatmap.py`:

- swap in success-rate `P_c`
- save raw arrays
- tighten plot formatting

### Step 3

Edit `/home/suresh/SHN/mixed-order/experiments/structured/structured_capacity.py`:

- make it a two-panel figure
- add multi-`D` sweep
- add drift panel
- save `g2` and `lambda^*`

### Step 4

Create `/home/suresh/SHN/mixed-order/experiments/learning/learning_boundary.py`:

- use the existing signal/drift measurement logic
- produce measured crossing-vs-`lambda`
- fit a single prefactor for the theory overlay

### Step 5

Create `/home/suresh/SHN/mixed-order/experiments/ham/xor3_hybrid_task.py`:

- synthetic task generator
- pairwise / cubic / mixed score evaluation
- accuracy-vs-noise plot

### Step 6

Move the old scripts out of the thesis narrative:

- keep `limiting_cases.py` only as a diagnostic
- remove `softmax_retrieval.py` from the thesis path
- replace `phase_diagram.py` in the thesis text with `learning_boundary.py`

## 10. Acceptance Criteria

The experiment layer is complete when all of the following are true.

### 10.1 Baseline

- the empirical heatmap qualitatively matches the theory heatmap
- the empirical optimum ridge tracks `lambda^*(p)`
- one single figure is enough to support Chapter 2

### 10.2 Structured

- the drift panel clearly shows uncentered drift and centered suppression
- the centered `P_c(lambda)` optimum shifts monotonically with measured `g2`
- one single figure is enough to support Chapter 3

### 10.3 Learning Boundary

- the measured crossing points are approximately linear in `1 + lambda/2`
- the figure reports the fitted law, not just raw curves
- one single figure is enough to support Chapter 4

### 10.4 HAM

- the synthetic task genuinely removes pairwise information from the parity block
- pairwise-only does not solve the higher-order part
- mixed outperforms pairwise-only over a nontrivial noise range
- the figure supports the Chapter 5 score-family argument without overselling it

## 11. What Not To Build

Do not build these unless a real failure forces it:

- a generic experiment framework
- a database or registry of runs
- multiple baseline sanity figures for the thesis
- several synthetic HAM tasks
- a full learned HAM training pipeline
- a custom plotting package

The current repository is small enough that flat scripts plus a few shared metric helpers are the right design.

## 12. Final Deliverable Set

After implementation, the thesis-facing experiment directory should effectively center on these four scripts:

- `/home/suresh/SHN/mixed-order/experiments/capacity_heatmap/capacity_heatmap.py`
- `/home/suresh/SHN/mixed-order/experiments/structured/structured_capacity.py`
- `/home/suresh/SHN/mixed-order/experiments/learning/learning_boundary.py`
- `/home/suresh/SHN/mixed-order/experiments/ham/xor3_hybrid_task.py`

That is the full experimental story. It matches the thesis chapters, it is minimal, and it avoids filler.
