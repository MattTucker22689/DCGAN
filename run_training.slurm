#!/bin/bash
#SBATCH --job-name=TrainingDCGAN                # create a short name for your job
#SBATCH --time=24:00:00                         # time #24:00:00 #
#SBATCH --nodes=1                               # node count
#SBATCH --ntasks=1                              # total number of tasks across all nodes
#SBATCH --cpus-per-task=1 #1                    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=25G                       # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1                            # request 1 gpu 


module purge
source /home/w223u672/run.sh

python /home/w223u672/AI_Projects/Music2Pix/DCGAN/dcgan_.py
