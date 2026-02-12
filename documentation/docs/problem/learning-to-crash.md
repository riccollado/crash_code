# Learning to Crash: Problem and Solution Discussion

This page provides a detailed statement of the optimization problem, the solution
strategy implemented in this repository, and how each supporting note/paper contributes
to the design.

## Problem statement

We solve a **stochastic project crashing** problem on a directed acyclic precedence
network.

Given:

- activities with uncertain durations,
- optional crash decisions with direct costs,
- precedence constraints,
- and completion-time penalties,

we seek a crashing plan that minimizes expected total cost under uncertainty.

### Inputs

1. Project network (GraphML/XML).
2. Three-point time estimates per activity (optimistic, most likely, pessimistic).
3. Correlation matrix for activity durations.
4. Penalty-function parameters (thresholds + shape).
5. Branching policy (`KG`, `Random`, `Uniform`, `Distance`, `Pareto_Inverse`,
   `Pareto_Boltzman`).
6. Scenario/sample budget.

### Outputs

- Best-found crashing solution and summary statistics.
- Optional experiment, iteration, and solution records in SQL tables.

## Uncertainty model and scenario generation

The uncertainty model follows a PERT-Beta + copula pipeline:

1. Convert three-point estimates to Beta-distribution parameters.
2. Build a valid correlation matrix for activities.
3. Sample from a multivariate normal with that correlation.
4. Transform to uniform marginals via normal CDF.
5. Apply inverse CDF of each activity's Beta distribution.

This produces scenario vectors where marginals match activity distributions and
cross-activity dependence is preserved.

Primary reference notes and papers:

- [Notes for Activity Crashing](../reference/notes_for_activity_crashing.md)
- [Project Simulation Using PERT-Beta Distributions (PDF)](../reference/papers/project_simulation_using_pert_beta_distributions.pdf)

## Optimization model (high level)

Each scenario evaluates a mixed-integer scheduling/crashing model with:

- binary crash-decision variables,
- precedence constraints on activity start times,
- penalty variables for piecewise penalty approximation,
- objective combining crash costs and lateness penalties.

The solution process operates on a **partition of subproblems**, where each subproblem
adds fixed decisions and receives stochastic bound estimates.

Implementation context:

- [Partitioning Algorithm Notes](../reference/partitioning_algorithm_notes.md)

## Branching and selection policies

The solver supports multiple branching policies with different exploration/exploitation
tradeoffs.

### Knowledge Gradient (`KG`)

Uses correlated normal-belief updates to select the next most informative leaf.

### Random / Uniform

- `Random`: stochastic leaf choice.
- `Uniform`: broad exploration across available leaves.

### Distance and Pareto-based selection

Uses non-dominated sorting in mean-variance space, then selects fronts by rank-aware
probabilities.

Pareto-Inverse probability:

$$
p_i = \frac{\left(1 / i\right)^\beta}{\sum_{k=1}^{n} \left(1 / k\right)^\beta}
$$

Pareto-Boltzman probability:

$$
p_i = \frac{e^{-\beta i}}{\sum_{k=1}^{n} s_k e^{-\beta k}}
$$

Normalization:

$$
\sum_{i=1}^{n} s_i p_i = 1
$$

Where:

- $\beta$: selection pressure,
- $n$: number of fronts,
- $s_i$: number of solutions in front $i$.

References:

- [Evolutionary Method](../reference/evolutionary_method.md)
- [Fast Nondominated Sorting Approach (PDF)](../reference/papers/fast_nondominated_sorting_approach.pdf)

## Stochastic branch-and-bound context

The core algorithm follows stochastic branch-and-bound principles:

- partition feasible decisions,
- estimate stochastic bounds on each partition,
- prioritize promising regions,
- refine until stopping conditions are met.

Primary theory links:

- [Stochastic Branch-and-Bound (PDF)](../reference/papers/stochastic_branch_and_bound.pdf)
- [Stochastic Integer Optimization (PDF)](../reference/papers/stochastic_integer_opt.pdf)

## Statistical confidence and reporting

Bootstrap-style resampling is used to quantify uncertainty of empirical statistics and
solution-quality summaries.

Reference:

- [Bootstrap (PDF)](../reference/papers/bootstrap.pdf)

## Full supporting library (all relevant files)

### Technical notes

- [Evolutionary Method](../reference/evolutionary_method.md): selection-pressure and
  Pareto-front probability models.
- [Notes for Activity Crashing](../reference/notes_for_activity_crashing.md): graph,
  distribution, correlation, and scenario-generation blueprint.
- [Partitioning Algorithm Notes](../reference/partitioning_algorithm_notes.md):
  subproblem structure, branching flow, and bound-estimation procedure.

### Papers

- [Bootstrap (PDF)](../reference/papers/bootstrap.pdf)
- [Crash Learning (PDF)](../reference/papers/crash_learning.pdf)
- [Fast Nondominated Sorting (PDF)](../reference/papers/fast_nondominated_sorting_approach.pdf)
- [Project Simulation with PERT-Beta (PDF)](../reference/papers/project_simulation_using_pert_beta_distributions.pdf)
- [Stochastic Branch-and-Bound (PDF)](../reference/papers/stochastic_branch_and_bound.pdf)
- [Stochastic Integer Optimization (PDF)](../reference/papers/stochastic_integer_opt.pdf)

For a compact index with rationale, see
[Research Library](../reference/related-literature.md).
