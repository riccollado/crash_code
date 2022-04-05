-- Table: public.crash_experiment

-- DROP TABLE public.crash_experiment;

CREATE TABLE public.crash_experiment
(
    id bigint NOT NULL DEFAULT nextval('crash_experiment_exp_id_seq'::regclass),
    network jsonb NOT NULL,
    no_of_edges integer,
    no_of_nodes integer,
    most_likely double precision[] NOT NULL,
    optimistic double precision[] NOT NULL,
    pessimistic double precision[] NOT NULL,
    cov_mat double precision[] NOT NULL,
    crash_cost double precision[] NOT NULL,
    crash_time double precision[] NOT NULL,
    penalty_b1 double precision NOT NULL,
    penalty_m double precision NOT NULL,
    penalty_steps integer NOT NULL,
    t_final double precision NOT NULL,
    t_init double precision NOT NULL,
    penalty_type text COLLATE pg_catalog."default" NOT NULL,
    kg_l double precision,
    kg_sigma double precision,
    kg_lambda jsonb,
    kg_mu jsonb,
    bootstrap boolean NOT NULL,
    confidence double precision,
    pareto_beta double precision,
    resamples integer,
    scenarios_per_estimation integer NOT NULL,
    total_scenarios integer NOT NULL,
    method_type text COLLATE pg_catalog."default" NOT NULL,
    seed integer,
    seed_np integer,
    network_figure bytea,
    network_pos jsonb,
    trial double precision[],
    CONSTRAINT crash_experiment_pkey PRIMARY KEY (id)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.crash_experiment
    OWNER to ricardo;
COMMENT ON TABLE public.crash_experiment
    IS 'Main table holding individual crash experiments.';

COMMENT ON COLUMN public.crash_experiment.network_figure
    IS 'Binary figure storage.';

COMMENT ON COLUMN public.crash_experiment.network_pos
    IS 'Network figure node positions. It can be used for visualizations.';
