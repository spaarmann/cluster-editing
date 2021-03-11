#!/usr/bin/env bash

set -e
set -u

INSTANCE_LIST="$1"

for INSTANCE in $INSTANCE_LIST
do
	sbatch ./slurm_submit.sh $INSTANCE
done
