#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=400G
#SBATCH --time=20:00:00
#SBATCH -e ./MPL/out/inf-old10-error
#SBATCH -o ./MPL/out/inf-old10-out
#SBATCH -p batch

module unload miniconda2
module load miniconda3
conda activate sars-env

outfile=/rhome/blee098/bigdata/SARS-CoV-2-Data/infer-new-2021-08-14-g-10
python epi-inf-parallel-old.py --data /rhome/blee098/bigdata/SARS-CoV-2-Data/2021-08-14/freqs --timed 1 -o $outfile -q 5 --g1 10 

