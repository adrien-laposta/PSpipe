#! /bin/bash
#
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=16       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH -t 6:00:00
#SBATCH -p physics


module load anaconda3
export OMP_NUM_THREADS=16
export JULIA_NUM_THREADS=16

$1
