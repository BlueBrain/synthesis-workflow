#!/bin/bash -l

export OMP_NUM_THREADS=1

python -m luigi --module synthesis_workflow.tasks.workflows ValidateVacuumSynthesis \
    --local-scheduler \
    --log-level INFO \
