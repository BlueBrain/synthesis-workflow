#!/bin/bash -l

python -m luigi --module synthesis_workflow.tasks.synthesis BuildSynthesisParameters \
    --local-scheduler \
    --log-level INFO \
