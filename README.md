Learning to Crash
==========

A Reinforcement Learning Approach to Project Scheduling
-------------------------------------------------------

This repository contains a CLI tool to implement a stochastic branch & bound optimization method for the problem of crashing a project activity network subject to uncertain activity times and threshold penalties.

Overview
--------

This application approximates a solution to the project management problem of finding an optimal crashing plan with uncertain activity times. The basic problem description can be found [here](https://pubsonline.informs.org/doi/pdf/10.1287/ijoc.12.2.125.11894?casa_token=PHCfqHAG120AAAAA:BsTfR2bDQEtx3tlkzJKbYcMAoSdDEcr65TkYU49hMCOUfULXn32p-9Li6bhKLWL-UpttA4DecBhA "A Stochastic Branch-and-Bound Approach to ActivityCrashing in Project Management").
The solution approach follows the stochastic branch & bound method presented in [here](https://pubsonline.informs.org/doi/pdf/10.1287/opre.46.3.381?casa_token=QsdLQM3thP0AAAAA:INj4Dv_NYAD48aM_odTL9AKv4dJHsbIguQSgHucoBmkDhPjoM5j8Z1kM16sZTXuANemOHEcp9kYT "On Optimal Allocation of Indivisibles Under Uncertainty") and applies the knowledge gradient technique introduced [here](https://pubsonline.informs.org/doi/pdf/10.1287/ijoc.1080.0314?casa_token=mADfuyTiLiMAAAAA:_NP3QhLLq_8ghTjK31heitjBhxa_YbEcEy0ng9QfaQlcGGtpusX7YrCMbfIarnGTNNQHHx76PJ9n "The Knowledge-Gradient Policy for COrrelated Normal Beliefs").

The problem requires the following input data:

1. A project network graph in the GraphML format (supplied as an XML file).

2. Lists of pessimistic, most likely, and optimistic, values to describing the PERT activity time distributions.

3. A covariance matrix describing the linear relationships among activity times.

4. Threshold values describing penalties due to time overrun.

5. A selection of branching decision method to apply from: "KG", "Random", "Uniform", "Distance", "Pareto\_Inverse", and "Pareto\_Boltzman".

6. Number of samples that the user wants to consider.

The output of the CLI is an XML file with the solution obtained after applying the stochastic branch & bound method with the selected branching decision.

The branching strategies are described below:

 1. **KG**: Branching selection via the knowledge gradient with correlated normal beliefs method discussed [here](https://pubsonline.informs.org/doi/pdf/10.1287/ijoc.1080.0314?casa_token=mADfuyTiLiMAAAAA:_NP3QhLLq_8ghTjK31heitjBhxa_YbEcEy0ng9QfaQlcGGtpusX7YrCMbfIarnGTNNQHHx76PJ9n "The Knowledge-Gradient Policy for COrrelated Normal Beliefs").

 2. **Random**: Random branching selection as discussed [here](https://pubsonline.informs.org/doi/pdf/10.1287/ijoc.12.2.125.11894?casa_token=PHCfqHAG120AAAAA:BsTfR2bDQEtx3tlkzJKbYcMAoSdDEcr65TkYU49hMCOUfULXn32p-9Li6bhKLWL-UpttA4DecBhA "A Stochastic Branch-and-Bound Approach to ActivityCrashing in Project Management").

 3. **Uniform**: Branching decision rule based on sampling on every available B&B leaf at every main iteration step.

 4. **Distance**: Branching decision obtained by forming the Pareto frontier of mean-variance pairs of each leaf and performing non-dominated sorting based on vector distance. The non-dominated sorting method is discussed [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=996017&casa_token=RX5FX8Ctu38AAAAA:BymQiux3DQammBgBVQANxxHhwDx5fhxT3FqRNB8nCvyND4WSajGqwvjyKNpISKO5aJj2akki&tag=1 "A Fast and Elitist Multiobjective Genetic Algorithm:").

 5. **Pareto\_Inverse**: Branching decision obtained by forming the Pareto frontier of mean-variance pairs of each leaf and performing non-dominated sorting based on inverse proportional probability:

<p>
<img src="https://latex.codecogs.com/gif.latex?p_i&space;=&space;\frac{\left(1/i\right)^\beta}{\sum_{k=1}^{n}\left(1/k\right)^\beta}" title="p_i = \frac{\left(1/i\right)^\beta}{\sum_{k=1}^{n}\left(1/k\right)^\beta}", />
</p>

<p>
where <img src="https://latex.codecogs.com/gif.latex?\beta" title="\beta" /> is the Pareto front selection pressure, <img src="https://latex.codecogs.com/gif.latex?n" title="n" /> is the total number of fronts, <img src="https://latex.codecogs.com/gif.latex?s_i" title="s_i" /> is the total number of solutions in front of rank <img src="https://latex.codecogs.com/gif.latex?i" title="i" />, and <img src="https://latex.codecogs.com/gif.latex?\inline&space;\sum_{i=1}^n&space;s_ip_i&space;=&space;1" title="\sum_{i=1}^n s_ip_i = 1" />.

6. **Pareto\_Boltzman**: Branching decision obtained by forming the Pareto frontier of mean-variance pairs of each leaf and performing non-dominated sorting based on inverse Boltzman probability:

<p>
<img src="https://latex.codecogs.com/gif.latex?p_i&space;=&space;\frac{e^{-\beta&space;i}}{\sum_{k=1}^n&space;s_k&space;e^{-\beta&space;k}}" title="p_i = \frac{e^{-\beta i}}{\sum_{k=1}^n s_k e^{-\beta k}}", />
</p>

where <img src="https://latex.codecogs.com/gif.latex?\beta" title="\beta" /> is the Pareto front selection pressure, <img src="https://latex.codecogs.com/gif.latex?n" title="n" /> is the total number of fronts, <img src="https://latex.codecogs.com/gif.latex?s_i" title="s_i" /> is the total number of solutions in front of rank <img src="https://latex.codecogs.com/gif.latex?i" title="i" />, and <img src="https://latex.codecogs.com/gif.latex?\inline&space;\sum_{i=1}^n&space;s_ip_i&space;=&space;1" title="\sum_{i=1}^n s_ip_i = 1" />.

The application also stores every solution and iteration in a SQL database for later retrieval and statistical analysis.

Dependencies
------------

- Python 3.9
- Numpy 1.22.3
- Scipy 1.8.0
- Pandas 1.4.2
- SQLAlchemy 1.4.34
- Matplotlib 3.5.1
- Gurobipy 9.5.1
- Pyfiglet 0.8.post1
- Bootstrapped 0.0.2
- Jellyfish 0.9.0
- Statsmodels 0.13.2
- Pydot 1.4.2
- Graphviz 0.19.1
- Networkx 2.7.1

Description of files
--------------------

Non-Python files:

| filename  | description                                             |
| --------- | ------------------------------------------------------- |
| README.md | Text file (markdown format) description of the project. |
| sql/*.sql | Postgres SQL files for database table creation          |
| output    | Output images (network and statistics).                 |

Python scripts files:

| filename        | description                                                                                          |
| --------------- | ---------------------------------------------------------------------------------------------------- |
| single_run.py   | Performs a single run of the method with specified parameters and randomly generated input data.     |
| multiple_run.py | Performs a multiple runs of the methods with specified parameters and randomly generated input data. |

Python modules:

| filename                  | description                                                                                                                                                            |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| generator_covariance.py   | Generates covariance matrices for the activity times.                                                                                                                  |
| generator_crash.py        | Generates activity crash times and costs.                                                                                                                              |
| generator_distribution.py | Generates PERT distributions for activity times.                                                                                                                       |
| generator_network.py      | Generates connected project network graphs.                                                                                                                            |
| generator_penalty.py      | Generates penalty linear and exponential values for the optimization problem.                                                                                          |
| generator_scenario.py     | Generates random samples based on activity times PERT and correlation matrix.                                                                                          |
| generator_subproblem.py   | Defines and solve with Gurobi an intermediate optimization problem with some fixed variables.                                                                          |
| knowledge_gradient.py     | Implements the Knowledge-Gradient algorithm with correlated beliefs.                                                                                                   |
| nearest_correlation.py    | Nick Higham's nearest correlation algorithm. Python implementation by Mike Croucher found [here](https://github.com/mikecroucher/nearest_correlation "Mike Croucher"). |
| optimize.py               | Commits problem to DB and calls the branch & bound method.                                                                                                             |
| stochastic.py             | Main file containing the branch & bound method and all supporting functions and methods. Includes methods to perform bootstrap and Pareto-based branching decision.    |
| uncrashed_bounds.py       | Solves the un-crashed and un-penalized scheduling problem on a single scenario to obtain bounds on costs and time.                                                     |
| utilities.py              | Numerical utility functions.                                                                                                                                           |
