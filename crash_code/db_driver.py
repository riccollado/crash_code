"""Database driver."""

import json
import os
import pickle

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


def initialize_db():
    """Initialize database.

    Returns:
        _type_: _description_
    """
    # Database connection from envirionment variables
    connect_url = URL(
        "postgres",
        host=os.environ["crash_db_host"],
        port=os.environ["crash_db_port"],
        username=os.environ["crash_db_user"],
        password=os.environ["crash_db_password"],
        database=os.environ["crash_db"],
    )

    # db = create_engine(connect_url)
    base = declarative_base()

    # Classs to manage meta experiments
    class Meta(base):
        __tablename__ = "crash_meta"
        id = Column(BIGINT, Sequence("crash_meta_id_seq"), primary_key=True)
        experiments = Column(ARRAY(BIGINT), nullable=False)

    # Class to manage single experiment data
    class Experiment(base):
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
    class Iteration(base):
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
    class Output(base):
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
    Session = sessionmaker(db)
    session = Session()
    base.metadata.create_all(db)

    # Variable used to hold current experiment id
    experiment_id = 0

    def push_experiment_db(seeds, attributes, run_time=0.0):
        """Push experiment to database.

        Args:
            seeds (_type_): _description_
            attributes (_type_): _description_
            run_time (float, optional): _description_. Defaults to 0.0.

        Returns:
            _type_: _description_
        """
        nonlocal experiment_id

        # Serialize the network
        G = attributes["network"]
        G_str = json.dumps(json_graph.node_link_data(G))

        experiment = Experiment(
            network=G_str,
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
        COV, partition_list, attributes, iteration_number, elapsed_iter_time
    ):
        """Push individual iteration in database.

        Args:
            COV (_type_): _description_
            partition_list (_type_): _description_
            attributes (_type_): _description_
            iteration_number (_type_): _description_
            elapsed_iter_time (_type_): _description_
        """
        iteration = Iteration(
            exp_id=experiment_id,
            cov=COV.to_json(compression="infer"),
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

    def update_exp_time(new_time):
        """Update experiment time.

        Args:
            new_time (_type_): _description_
        """
        experiment = session.query(Experiment).get(experiment_id)
        experiment.exp_time = new_time
        session.commit()
        return

    def push_solution_db(solution):
        """Push solution to database.

        Args:
            solution (_type_): _description_
        """
        solution = Output(
            exp_id=experiment_id,
            e_sol=solution["E_solution"],
            e_data=solution["E_data"],
            std_sol=solution["Std_sol"],
            std_data=solution["Std_data"],
            partial_sol=json.dumps(solution["Partial_sol"]),
        )
        # Commit iteration
        session.add(solution)
        session.commit()
        return

    def close_db():
        # Close session
        session.close()
        return

    return (
        push_experiment_db,
        push_iteration_db,
        update_exp_time,
        push_solution_db,
        close_db,
    )
