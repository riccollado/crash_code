CREATE TABLE public.crash_meta
(
    id bigserial,
    experiments bigint[] NOT NULL
)
WITH (
    OIDS = FALSE
);

ALTER TABLE public.crash_meta
    OWNER to ricardo;
COMMENT ON TABLE public.crash_meta
    IS 'This table holds links single experiments created by multiple-experiment runs. In this way, we can reference all runs that we performed in a meta-experiment and apply statistics to it later on.';

COMMENT ON COLUMN public.crash_meta.experiments
    IS 'List of experiment ids of all experiments performed in this meta-experiment run.';
