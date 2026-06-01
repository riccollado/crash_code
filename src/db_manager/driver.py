"""PostgreSQL persistence layer for crash experiments.

Defines SQLAlchemy ORM models (Meta, Experiment, Iteration, Output) and
exposes a factory function that returns five closures for writing
experiment data to the database.
"""

import json
import os
import pickle
from typing import Any, Callable

import pandas as pd
from networkx.readwrite import json_graph
from sqlalchemy import URL, Column, ForeignKey, Sequence, create_engine
from sqlalchemy.dialects.postgresql import (
    ARRAY,
    BIGINT,
    BOOLEAN,
    BYTEA,
    DOUBLE_PRECISION,
    INTEGER,
    JSONB,
    TEXT,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker


def initialize_db() -> tuple[  # pylint: disable=too-many-statements
    Callable,
    Callable,
    Callable,
    Callable,
    Callable,
]:
    """Initialize the database and return functions to interact with it.

    This function sets up the database connection, defines the necessary
    SQLAlchemy models, and creates the database tables if they do not exist.
    It returns a set of functions to interact with the database, including
    pushing experiments, pushing iterations, updating experiment time, pushing
    solutions, and closing the database session.

    Returns
    -------
    tuple
        A tuple containing the following functions:
        - push_experiment_db: Function to push an experiment to the database.
        - push_iteration_db: Function to push an iteration to the database.
        - update_exp_time: Function to update the experiment time.
        - push_solution_db: Function to push a solution to the database.
        - close_db: Function to close the database session.
    """
    # Database connection from environment variables
    connect_url = URL.create(
        "postgresql",
        host=os.environ["crash_db_host"],
        port=os.environ["crash_db_port"],
        username=os.environ["crash_db_user"],
        password=os.environ["crash_db_password"],
        database=os.environ["crash_db"],
    )

    # db = create_engine(connect_url)
    Base: type = declarative_base()  # pylint: disable=invalid-name

    class Meta(Base):  # pylint: disable=unused-variable
        """Metadata grouping for crash experiments.

        Attributes
        ----------
        id : Column(BIGINT)
            Primary key (auto-sequenced).
        experiments : Column(ARRAY(BIGINT))
            Array of experiment IDs belonging to this group.
        """

        __tablename__ = "crash_meta"
        id = Column(BIGINT, Sequence("crash_meta_id_seq"), primary_key=True)
        experiments = Column(ARRAY(BIGINT), nullable=False)

    class Experiment(Base):
        """Single crash experiment with full problem definition and results.

        Stores the project network, PERT parameters, crash alternatives,
        penalty configuration, method settings, and elapsed time.  Has
        one-to-many relationships to ``Iteration`` and ``Output``.
        """

        __tablename__ = "crash_experiment"

        iterations = relationship("Iteration")
        results = relationship("Output")
        id = Column(BIGINT, Sequence("crash_experiment_id_seq"), primary_key=True)

        network = Column(JSONB, nullable=False)
        no_of_edges = Column(INTEGER)
        no_of_nodes = Column(INTEGER)

        most_likely = Column(ARRAY(DOUBLE_PRECISION), nullable=False)
        optimistic = Column(ARRAY(DOUBLE_PRECISION), nullable=False)
        pessimistic = Column(ARRAY(DOUBLE_PRECISION), nullable=False)
        cov_mat = Column(BYTEA, nullable=False)

        crash_cost = Column(ARRAY(DOUBLE_PRECISION), nullable=False)
        crash_time = Column(ARRAY(DOUBLE_PRECISION), nullable=False)

        penalty_b1 = Column(DOUBLE_PRECISION, nullable=False)
        penalty_m = Column(DOUBLE_PRECISION, nullable=False)
        penalty_steps = Column(INTEGER, nullable=False)
        t_final = Column(DOUBLE_PRECISION, nullable=False)
        t_init = Column(DOUBLE_PRECISION, nullable=False)
        penalty_type = Column(TEXT, nullable=False)

        kg_l = Column(DOUBLE_PRECISION)
        kg_sigma = Column(DOUBLE_PRECISION)
        kg_lambda = Column(JSONB)
        kg_mu = Column(JSONB)

        bootstrap = Column(BOOLEAN, nullable=False)
        confidence = Column(DOUBLE_PRECISION)
        resamples = Column(INTEGER)

        pareto_beta = Column(DOUBLE_PRECISION)

        scenarios_per_estimation = Column(INTEGER, nullable=False)
        total_scenarios = Column(INTEGER, nullable=False)
        method_type = Column(TEXT, nullable=False)

        seed = Column(INTEGER)
        seed_np = Column(INTEGER)

        network_figure = Column(BYTEA)
        network_pos = Column(JSONB)

        exp_time = Column(DOUBLE_PRECISION)

    class Iteration(Base):
        """Snapshot of the B&B tree state after one SB&B iteration.

        Stores the KG belief vectors, constraint tree, per-leaf bound
        estimates, and wall-clock time for a single iteration.
        """

        __tablename__ = "crash_iteration"

        id = Column(BIGINT, Sequence("crash_iteration_id_seq"), primary_key=True)
        exp_id = Column(BIGINT, ForeignKey("crash_experiment.id"), nullable=False)

        cov = Column(JSONB)
        kg_mu = Column(ARRAY(DOUBLE_PRECISION))
        kg_lambda = Column(ARRAY(DOUBLE_PRECISION))
        constr_tree = Column(JSONB)
        kg_e_tree = Column(ARRAY(DOUBLE_PRECISION))
        e_tree = Column(ARRAY(DOUBLE_PRECISION))
        std_tree = Column(ARRAY(DOUBLE_PRECISION))
        recordset_tree = Column(ARRAY(BOOLEAN))
        singleton_tree = Column(ARRAY(BOOLEAN))
        iteration_num = Column(BIGINT, nullable=False)
        iter_time = Column(DOUBLE_PRECISION)

    class Output(Base):
        """Final or intermediate solution produced by the optimizer.

        Stores the expected cost, per-leaf cost array, standard deviation,
        and the partial crashing solution as JSON.
        """

        __tablename__ = "crash_output"

        id = Column(BIGINT, Sequence("crash_output_id_seq"), primary_key=True)
        exp_id = Column(BIGINT, ForeignKey("crash_experiment.id"), nullable=False)

        e_sol = Column(DOUBLE_PRECISION, nullable=False)
        e_data = Column(ARRAY(DOUBLE_PRECISION), nullable=False)
        std_sol = Column(DOUBLE_PRECISION, nullable=False)
        std_data = Column(ARRAY(DOUBLE_PRECISION), nullable=False)
        partial_sol = Column(JSONB, nullable=False)

    # Create session
    db = create_engine(connect_url)
    db_session = sessionmaker(db)
    session = db_session()
    Base.metadata.create_all(db)

    # Variable used to hold current experiment id
    experiment_id = 0

    def push_experiment_db(
        seeds: list[int],
        attributes: dict[str, Any],
        run_time: float = 0.0,
    ) -> int:
        """Serialize and insert a new experiment row.

        Parameters
        ----------
        seeds : list of int
            Two-element list ``[general_seed, numpy_seed]``.
        attributes : dict
            Problem and method attributes produced by
            ``initialize_attributes``.
        run_time : float, optional
            Elapsed wall-clock time in seconds (default 0.0).

        Returns
        -------
        int
            Primary-key ID of the newly created experiment row.
        """
        nonlocal experiment_id
        nonlocal session

        # Serialize the network
        network_graph = attributes["network"]
        network_graph_serialized = json.dumps(json_graph.node_link_data(network_graph))

        experiment = Experiment(
            network=network_graph_serialized,
            no_of_edges=attributes["network"].number_of_edges(),
            no_of_nodes=attributes["network"].number_of_nodes(),
            optimistic=attributes["optimistic"],
            most_likely=attributes["most_likely"],
            pessimistic=attributes["pessimistic"],
            cov_mat=pickle.dumps(attributes["cov_mat"]),
            crash_cost=attributes["crash_cost"],
            crash_time=attributes["crash_time"],
            penalty_b1=attributes["b1"],
            penalty_m=attributes["m"],
            penalty_steps=attributes["penalty_steps"],
            t_final=attributes["t_final"],
            t_init=attributes["t_init"],
            penalty_type=attributes["penalty_type"],
            kg_l=attributes["KG_l"],
            kg_sigma=attributes["KG_sigma"],
            kg_lambda=attributes["KG_lambda"],
            kg_mu=attributes["KG_mu"],
            bootstrap=attributes["bootstrap"],
            confidence=attributes["confidence"],
            resamples=attributes["resamples"],
            pareto_beta=attributes["pareto_beta"],
            scenarios_per_estimation=attributes["scen_est_num"],
            total_scenarios=attributes["total_scenarios"],
            method_type=attributes["method_type"],
            seed=seeds[0],
            seed_np=seeds[1],
            network_figure=attributes["binary_figure"],
            network_pos=json.dumps(attributes["pos_figure"]),
            exp_time=run_time,
        )

        # Commit experiment
        session.add(experiment)
        session.commit()

        # Remove pdf figure from memory
        attributes["binary_figure"] = []

        # Set experiment_id
        experiment_id = experiment.id

        return experiment_id

    def push_iteration_db(
        cov_matrix: pd.DataFrame,
        partition_list: list[dict[str, Any]],
        attributes: dict[str, Any],
        iteration_number: int,
        elapsed_iter_time: float,
    ) -> None:
        """Push individual iteration in database.

        Parameters
        ----------
        cov_matrix : pandas.DataFrame
            Covariance matrix of the current iteration.
        partition_list : list of dict
            List of dictionaries containing partition information.
        attributes : dict
            Dictionary containing various attributes related to the iteration.
        iteration_number : int
            The current iteration number.
        elapsed_iter_time : float
            The time taken for the current iteration.

        Returns
        -------
        None
        """
        nonlocal experiment_id
        nonlocal session

        iteration = Iteration(
            exp_id=experiment_id,
            cov=cov_matrix.to_json(compression="infer"),
            kg_mu=[attributes["KG_mu"][x] for x in attributes["KG_mu"].keys()],
            kg_lambda=[
                attributes["KG_lambda"][x] for x in attributes["KG_lambda"].keys()
            ],
            constr_tree=json.dumps([p["constraints"] for p in partition_list]),
            kg_e_tree=[p["KG_E"] for p in partition_list],
            e_tree=[p["E"] for p in partition_list],
            std_tree=[p["STD"] for p in partition_list],
            recordset_tree=[p["recordset"] for p in partition_list],
            singleton_tree=[p["singleton"] for p in partition_list],
            iteration_num=iteration_number,
            iter_time=elapsed_iter_time,
        )
        # Commit iteration
        session.add(iteration)
        session.commit()

        return

    def update_exp_time(new_time: float) -> None:
        """Overwrite the elapsed time on the current experiment row.

        Parameters
        ----------
        new_time : float
            New wall-clock time in seconds.

        Returns
        -------
        None
        """
        nonlocal experiment_id
        nonlocal session

        experiment = session.get(Experiment, experiment_id)
        experiment.exp_time = new_time
        session.commit()
        return

    def push_solution_db(solution: dict[str, Any]) -> None:
        """Insert a solution row for the current experiment.

        Parameters
        ----------
        solution : dict
            Keys: ``"E_solution"`` (float), ``"E_data"`` (list of float),
            ``"Std_sol"`` (float), ``"Std_data"`` (list of float),
            ``"Partial_sol"`` (dict, JSON-serialisable).

        Returns
        -------
        None
        """
        nonlocal experiment_id
        nonlocal session

        output_instance = Output(
            exp_id=experiment_id,
            e_sol=solution["E_solution"],
            e_data=solution["E_data"],
            std_sol=solution["Std_sol"],
            std_data=solution["Std_data"],
            partial_sol=json.dumps(solution["Partial_sol"]),
        )
        # Commit solution
        session.add(output_instance)
        session.commit()
        return

    def close_db() -> None:
        """Close the SQLAlchemy session.

        Returns
        -------
        None
        """
        nonlocal session
        session.close()
        return

    return (
        push_experiment_db,
        push_iteration_db,
        update_exp_time,
        push_solution_db,
        close_db,
    )
