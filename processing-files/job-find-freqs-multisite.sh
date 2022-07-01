#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=blee098@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH -e ./MPL/out/freq-error-%a
#SBATCH -o ./MPL/out/freq-out-%a
#SBATCH --array=0-160

files=(/rhome/blee098/bigdata/SARS-CoV-2-Data/data/freq_0.05/*)
module purge
module load anaconda3


cd ./Archive-vector
g++ src/main.cpp src/inf.cpp src/io.cpp -O3 -march=native -lgslcblas -lgsl -o bin/mpl -std=c++11
pwd

file=${files[$SLURM_ARRAY_TASK_ID]}
tempout=/rhome/blee098/bigdata/SARS-CoV-2-Data/2021-08-14/freqs
python ../epi-covar-parallel-old.py --data "$file" -o $tempout -q 5 --pop_size 10000 -k 0.1 -R 2 --scratch /scratch/freqs --timed 1

