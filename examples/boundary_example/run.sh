#!/bin/bash -l

export OMP_NUM_THREADS=1
export REGION_GROWER_BOUNDARY_DEBUG=1

rm -rf atlas
brainbuilder atlases -n 2,1 -t 100,100 -d 10 -o atlas column -a 200


python -m luigi --module synthesis_workflow.tasks.workflows ValidateSynthesis \
    --local-scheduler \
    --log-level INFO \
