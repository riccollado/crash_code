"""Experiment execution for project crashing optimization.

Provides the top-level entry point that generates a random problem instance,
configures the stochastic branch-and-bound method, and runs the full
optimization pipeline.

Modules
-------
single_run
    Main executable script (``python -m run_manager.single_run``).
    Generates a random project network, PERT distributions, crash
    alternatives, and penalty parameters, then invokes the optimizer and
    prints the results.
"""

__all__ = ["single_run"]
