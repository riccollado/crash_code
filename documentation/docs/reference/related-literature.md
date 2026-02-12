# Research Library

This section consolidates the project notes and foundational papers that inform the
problem formulation, scenario generation model, branching logic, and uncertainty
quantification strategy used in *Learning to Crash*.

## Technical notes

| File | Why it matters |
| --- | --- |
| [Evolutionary Method](evolutionary_method.md) | Defines the Pareto-front selection probabilities used by evolutionary-style branching policies (`Pareto_Inverse`, `Pareto_Boltzman`), including the role of selection pressure `beta`. |
| [Notes for Activity Crashing](notes_for_activity_crashing.md) | End-to-end design notes for generating project graphs, PERT-Beta marginals, correlation matrices, and Gaussian-copula scenarios. This is the most complete operational blueprint for input generation. |
| [Partitioning Algorithm Notes](partitioning_algorithm_notes.md) | Documents the partition/subproblem data model, branching operations, bound-estimation flow, and implementation considerations for the branch-and-bound routine. |

## Foundational papers

| Paper | Relevance to this project |
| --- | --- |
| [A Stochastic Branch-and-Bound Approach to Activity Crashing in Project Management](papers/stochastic_branch_and_bound.pdf) | Core methodological basis for stochastic branch-and-bound under uncertain activity durations. |
| [On Optimal Allocation of Indivisibles Under Uncertainty](papers/stochastic_integer_opt.pdf) | Theoretical foundation for stochastic discrete optimization and bound-driven search strategies. |
| [A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II](papers/fast_nondominated_sorting_approach.pdf) | Basis for non-dominated sorting and Pareto-front ranking behavior used in alternative selection rules. |
| [Teaching Project Simulation in Excel Using PERT-Beta Distributions](papers/project_simulation_using_pert_beta_distributions.pdf) | Practical formulas for mapping `(optimistic, most likely, pessimistic)` estimates into Beta parameters. |
| [An Introduction to Bootstrap](papers/bootstrap.pdf) | Resampling background for uncertainty quantification and statistical confidence estimates. |
| [Crash Learning Presentation](papers/crash_learning.pdf) | Project context, motivation, and communication-oriented summary of the approach. |

## How these artifacts map to the implementation

- **Input generation** (`src/gen_manager/*`): grounded by the activity-crashing and
  PERT-Beta notes/papers.
- **Optimization core** (`src/opt_manager/*`): grounded by stochastic
  branch-and-bound and stochastic integer optimization references.
- **Pareto-based branching** (`Distance`, `Pareto_*`): grounded by NSGA-II and
  evolutionary-method notes.
- **Result uncertainty reporting**: informed by bootstrap notes and supporting papers.
