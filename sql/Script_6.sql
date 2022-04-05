CREATE TABLE public.crash_experiment
(
    id bigserial,
    network jsonb NOT NULL,
    no_of_edges integer,
    no_of_nodes integer,
    most_likely double precision[] NOT NULL,
    optimistic double precision[] NOT NULL,
    pessimistic double precision[] NOT NULL,
    cov_mat jsonb NOT NULL,
    crash_cost double precision[] NOT NULL,
    crash_time double precision[] NOT NULL,
    penalty_b1 double precision NOT NULL,
    penalty_m double precision NOT NULL,
    penalty_steps integer NOT NULL,
    t_final double precision NOT NULL,
    t_init double precision NOT NULL,
    penalty_type text NOT NULL,
    kg_l double precision,
    kg_sigma double precision,
    kg_lambda jsonb,
    kg_mu jsonb,
    bootstrap boolean NOT NULL,
    confidence double precision,
    resamples integer,
    pareto_beta double precision,
    scenarios_per_estimation integer NOT NULL,
    total_scenarios integer NOT NULL,
    method_type text NOT NULL,
    seed integer,
    seed_np integer,
    network_figure bytea,
    network_pos jsonb,
    CONSTRAINT experiment_id PRIMARY KEY (id)
)
WITH (
    OIDS = FALSE
);

ALTER TABLE public.crash_experiment
    OWNER to ricardo;
COMMENT ON TABLE public.crash_experiment
    IS 'Main table holding individual crash experiments.';

COMMENT ON COLUMN public.crash_experiment.network_figure
    IS 'Binary figure storage.';

COMMENT ON COLUMN public.crash_experiment.network_pos
    IS 'Network figure node positions. It can be used for visualizations.';
