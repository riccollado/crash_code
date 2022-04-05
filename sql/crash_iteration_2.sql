CREATE TABLE public.crash_iteration
(
    id bigserial,
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
    PRIMARY KEY (id),
    CONSTRAINT "Experiment_id" FOREIGN KEY (exp_id)
        REFERENCES public.crash_experiment (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
        NOT VALID
)
WITH (
    OIDS = FALSE
);

ALTER TABLE public.crash_iteration
    OWNER to ricardo;
COMMENT ON TABLE public.crash_iteration
    IS 'Table holding iteration data for crash_experiments. The column exp_id links the table to the corresponding experiment.';

COMMENT ON CONSTRAINT "Experiment_id" ON public.crash_iteration
    IS 'Link to experiment from which the iteration forms part of.';
