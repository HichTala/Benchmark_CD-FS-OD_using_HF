#!/bin/bash
#SBATCH --job-name={job_name}        # name of job
#SBATCH -A mvq@v100
# Other partitions are usable by activating/uncommenting
# one of the 5 following directives:
## SBATCH -C v100-16g                 # uncomment to target only 16GB V100 GPU
##SBATCH -C v100-32g                 # uncomment to target only 32GB V100 GPU
#SBATCH --partition=gpu_p2           # uncomment for gpu_p2 partition (32GB V100 GPU)
##SBATCH --partition=gpu_p4          # uncomment for gpu_p4 partition (40GB A100 GPU)
##SBATCH -C a100                     # uncomment for gpu_p5 partition (80GB A100 GPU)
# Here, reservation of 10 CPUs (for 1 task) and 1 GPU on a single node:
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 8 with gpu_p2, gpu_p4, gpu_p5)
# The number of CPUs per task must be adapted according to the partition used. Knowing that here
# only one GPU is reserved (i.e. 1/4 or 1/8 of the GPUs of the node depending on the partition),
# the ideal is to reserve 1/4 or 1/8 of the CPUs of the node for the single task:
##SBATCH --cpus-per-task=10           # number of cores per task (1/4 of the 4-GPUs node)
#SBATCH --cpus-per-task=3           # number of cores per task for gpu_p2 (1/8 of 8-GPUs node)
##SBATCH --cpus-per-task=6           # number of cores per task for gpu_p4 (1/8 of 8-GPUs node)
##SBATCH --cpus-per-task=8           # number of cores per task for gpu_p5 (1/8 of 8-GPUs node)
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=20:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=logs/gpu_single%j.out    # name of output file
#SBATCH --error=logs/gpu_single%j.out     # name of error file (here, in common with the output file)
#SBATCH --qos=qos_gpu-t3

module purge # purge modules inherited by default
module load pytorch-gpu/py3/2.5.0 # load modules

export PYTHONUSERBASE=$WORK/.local
export WANDB_MODE=offline
export NO_ALBUMENTATIONS_UPDATE=1
set -x # activate echo of launched commands

srun python -u run_object_detection.py{command}