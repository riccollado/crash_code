"""Database persistence layer for crash experiments.

Provides SQLAlchemy-backed storage of experiment definitions, per-iteration
state, and final solutions in a PostgreSQL database.

Modules
-------
driver
    Initializes the database connection and returns callable helpers to push
    experiments, iterations, and solutions, update timing, and close the
    session.

Examples
--------
>>> from db_manager.driver import initialize_db
>>> push_exp, push_iter, update_time, push_sol, close = initialize_db()
"""

__all__ = ["driver"]
