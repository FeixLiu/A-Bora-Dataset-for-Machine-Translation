#!/bin/bash -l

#$ -P statnlp
#$ -l h_rt=02:00:00
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=4
#$ -m ea

module load python3/3.6.5
module load gcc/5.5.0
module load cuda/9.2
module load pytorch/1.0
./run.sh
