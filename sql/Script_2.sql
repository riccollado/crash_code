CREATE TABLE public.crash_output
(
    id bigserial,
    exp_id bigint,
    CONSTRAINT output_id PRIMARY KEY (id),
    CONSTRAINT experiment_id FOREIGN KEY (exp_id)
        REFERENCES public.crash_experiment (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
        NOT VALID
)
WITH (
    OIDS = FALSE
);

ALTER TABLE public.crash_output
    OWNER to ricardo;
COMMENT ON TABLE public.crash_output
    IS 'Table to hold the output from crash optimization experiments.';
COMMENT ON CONSTRAINT output_id ON public.crash_output
    IS 'Unique Id of experiments output.';

COMMENT ON CONSTRAINT experiment_id ON public.crash_output
    IS 'References the experiment from where the result comes from.';
