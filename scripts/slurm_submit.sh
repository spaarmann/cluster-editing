#!/usr/bin/env bash
# 2020 Sebastian Paarmann
#SBATCH --partition=ether	 				    # the partition, `ether` has gigabit-ethernet for SMP, `ib` has infiniband for MPP
#       --qos=small                             # This was in Jens' script, but produces an error for some reason. (even though the qos exists)
#SBATCH --job-name=cluster-editing				# job name
#SBATCH --mail-user=sebastian.paarmann@tuhh.de	# where to send mail notifications
#       --mail-type=ALL							# ALL is useful for debugging,
#SBATCH --mail-type=END,FAIL,REQUEUE    		# but a bit noisy when starting 80+ jobs at once. Still nice to be notified when something finishes or crashes though.
#SBATCH --cpus-per-task=1						# `cluster-editing` is currently single-threaded, and we submit one job for a single instance each, so we only need one CPU/thread
#SBATCH --mem-per-cpu=1000        				# With the current non-cloning branching we need very little memory
#SBATCH --time=7-00:00:00						# days-hh:mm:ss time limit
#SBATCH	--export=ALL                            # Propagate complete environment to job 
#SBATCH --exclude=d[001-016]                    # excludes the nodes in the list (ranges may be used) from allocation; TUHH specific reason: because d0?? does not support the intrinsics of the other nodes and n0?? has bash problems (taken from HPC batch script by Jens M. Schmidt)

# Assumed folder layout:
# <current working dir when calling sbatch>
# |- `cluster-editing` executable
# |- `slurm_submit.sh`
# |- <the instances>, e.g. `exact001.gr`

# Call with e.g. `sbatch ./slurm_submit.sh exact001.gr` (or via `slurm_submit_all.sh`)

set -e
set -u

INSTANCE=$1

# The RZT website has an example copying things to the local filesystem like this,
# but in practice this seems to randomly fail on some jobs.
# Since we're not really IO-bound in any form, let's just use the current directory
# mounted over the network.

## Create a work dir on the local node
#MYWORKDIR=/usertemp/auk/crm0569/$SLURM_JOBID
#mkdir $MYWORKDIR

#cp $SLURM_SUBMIT_DIR/cluster-editing $MYWORKDIR
#cp $SLURM_SUBMIT_DIR/$INSTANCE $MYWORKDIR
#
#cd $MYWORKDIR

RUST_LOG=info RUST_BACKTRACE=1 ./cluster-editing $INSTANCE 2>&1 | tee $INSTANCE.out

#cp $INSTANCE.out $SLURM_SUBMIT_DIR/$INSTANCE.out

#cd ..
#rm -rf $MYWORKDIR

exit
