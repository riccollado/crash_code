"""Database driver."""

import json
import os
import pickle
from typing import Any, Callable, Dict, List, Tuple, Type

import pandas as pd
from networkx.readwrite import json_graph
from sqlalchemy import Column, ForeignKey, Sequence, create_engine
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
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker


def initialize_db() -> Tuple[  # pylint: disable=too-many-statements
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
    connect_url = URL(
        "postgres",
        host=os.environ["crash_db_host"],
        port=os.environ["crash_db_port"],
        username=os.environ["crash_db_user"],
        password=os.environ["crash_db_password"],
        database=os.environ["crash_db"],
    )

    # db = create_engine(connect_url)
    Base: Type = declarative_base()  # pylint: disable=invalid-name

    class Meta(Base):  # pylint: disable=unused-variable
        """
        Meta class representing the metadata for crash experiments.

        This class is used to store metadata for crash experiments, including
        the experiments associated with the metadata. In other words, this class
        is used to manage the meta experiments.

        Attributes:
            __tablename__ (str): The name of the table in the database.
            id (Column): The primary key for the table, using a sequence for unique
            values.
            experiments (Column): An array of BIGINTs representing the experiments
            associated with the metadata.
        """

        __tablename__ = "crash_meta"
        id = Column(BIGINT, Sequence("crash_meta_id_seq"), primary_key=True)
        experiments = Column(ARRAY(BIGINT), nullable=False)

    # Class to manage single experiment data
    class Experiment(Base):
        """
        Represents an experiment in the crash code database.

        Attributes:
            iterations (relationship): Relationship to the Iteration model.
            results (relationship): Relationship to the Output model.
            id (Column): Primary key for the experiment.
            network (Column): JSONB column representing the network.
            no_of_edges (Column): Number of edges in the network.
            no_of_nodes (Column): Number of nodes in the network.
            most_likely (Column): Array of most likely values.
            optimistic (Column): Array of optimistic values.
            pessimistic (Column): Array of pessimistic values.
            cov_mat (Column): Covariance matrix in BYTEA format.
            crash_cost (Column): Array of crash costs.
            crash_time (Column): Array of crash times.
            penalty_b1 (Column): Penalty b1 value.
            penalty_m (Column): Penalty m value.
            penalty_steps (Column): Number of penalty steps.
            t_final (Column): Final time value.
            t_init (Column): Initial time value.
            penalty_type (Column): Type of penalty.
            kg_l (Column): KG l value.
            kg_sigma (Column): KG sigma value.
            kg_lambda (Column): KG lambda value in JSONB format.
            kg_mu (Column): KG mu value in JSONB format.
            bootstrap (Column): Boolean indicating if bootstrap is used.
            confidence (Column): Confidence value.
            resamples (Column): Number of resamples.
            pareto_beta (Column): Pareto beta value.
            scenarios_per_estimation (Column): Number of scenarios per estimation.
            total_scenarios (Column): Total number of scenarios.
            method_type (Column): Type of method used.
            seed (Column): Seed value.
            seed_np (Column): Seed value for numpy.
            network_figure (Column): Network figure in BYTEA format.
            network_pos (Column): Network position in JSONB format.
            exp_time (Column): Experiment time.
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

    # Class to manage iteration data
    class Iteration(Base):
        """
        Represents an iteration in the crash experiment.

        Attributes:
            id (BIGINT): Primary key for the iteration.
            exp_id (BIGINT): Foreign key referencing the crash_experiment table.
            cov (JSONB): Covariance data.
            kg_mu (ARRAY of DOUBLE_PRECISION): Knowledge gradient mean values.
            kg_lambda (ARRAY of DOUBLE_PRECISION): Knowledge gradient lambda values.
            constr_tree (JSONB): Constraint tree data.
            kg_e_tree (ARRAY of DOUBLE_PRECISION): Knowledge gradient e-tree values.
            e_tree (ARRAY of DOUBLE_PRECISION): E-tree values.
            std_tree (ARRAY of DOUBLE_PRECISION): Standard deviation tree values.
            recordset_tree (ARRAY of BOOLEAN): Recordset tree data.
            singleton_tree (ARRAY of BOOLEAN): Singleton tree data.
            iteration_num (BIGINT): The iteration number.
            iter_time (DOUBLE_PRECISION): The time taken for the iteration.
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

    # Class to manage output data
    class Output(Base):
        """
        Represents the output of a crash experiment.

        Attributes:
            id (int): Primary key, unique identifier for the output.
            exp_id (int): Foreign key referencing the crash experiment.
            e_sol (float): Solution energy value.
            e_data (list of float): Array of energy data values.
            std_sol (float): Standard deviation of the solution.
            std_data (list of float): Array of standard deviation data values.
            partial_sol (dict): JSONB field containing partial solution data.
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
        seeds: List[int],
        attributes: Dict[str, Any],
        run_time: float = 0.0,
    ) -> int:
        """Push experiment to database.

        Args:
            seeds (List[int]): A list containing two seed values, one for general use
            and one for numpy.
            attributes (Dict[str, Any]): List of attributes related to the experiment.
            run_time (float): The time taken to run the experiment. Defaults to 0.0.

        Returns:
            int: The ID of the newly created experiment.
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
        partition_list: List[Dict[str, Any]],
        attributes: Dict[str, Any],
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
        """Update experiment time.

        Args:
            new_time (float): The new experiment time to be updated.
        """
        nonlocal experiment_id
        nonlocal session

        experiment = session.query(Experiment).get(experiment_id)
        experiment.exp_time = new_time
        session.commit()
        return

    def push_solution_db(solution: Dict[str, Any]) -> None:
        """Push solution to database.

        Args:
            solution (Dict[str, Any]): A dictionary containing the solution data with
            keys:
                - "E_solution" (float): Solution energy value.
                - "E_data" (List[float]): Array of energy data values.
                - "Std_sol" (float): Standard deviation of the solution.
                - "Std_data" (List[float]): Array of standard deviation data values.
                - "Partial_sol" (Dict[str, Any]): Partial solution data in JSON
                  serializable format.
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

    def close_db():
        """Close session."""
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
