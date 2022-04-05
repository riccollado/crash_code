-- Table: public.crash_iteration

-- DROP TABLE public.crash_iteration;

CREATE TABLE public.crash_iteration
(
    id bigint NOT NULL DEFAULT nextval('crash_iteration_id_seq'::regclass),
    exp_id bigint NOT NULL,
    cov jsonb,
    kg_mu double precision[],
    kg_lambda double precision[],
    constr_tree jsonb,
    kg_e_tree double precision[],
    e_tree double precision[],
    std_tree double precision[],
    recordset_tree boolean[],
    singleton_tree boolean[],
    CONSTRAINT "Iteration_id" PRIMARY KEY (id),
    CONSTRAINT "Experiment_id" FOREIGN KEY (exp_id)
        REFERENCES public.crash_experiment (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.crash_iteration
    OWNER to ricardo;
COMMENT ON TABLE public.crash_iteration
    IS 'Table holding iteration data for crash_experiments. The column exp_id links the table to the corresponding experiment.';

COMMENT ON CONSTRAINT "Experiment_id" ON public.crash_iteration
    IS 'Link to experiment from which the iteration forms part of.';
