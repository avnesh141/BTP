#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --time=04:00:00
#SBATCH --job-name=DNN_surgery
#SBATCH --error=DNN_Err.txt
#SBATCH --output=DNN_output.txt
#SBATCH --partition=cpu

python file.py