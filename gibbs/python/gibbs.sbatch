#!/bin/sh
#SBATCH -a 1-10
#SBATCH -p standard -N 1 --mem=62GB -t 36:00:00 -J motifs_with_start.%a -o rainier_motifs_with_starts.%a.log -e rainier_motifs_with_starts.%a.error --mail-type=all

module load python3 boost/1.65.0/b2 cmake 
DATADIR=/home/rbarret8/pymc3_qspr
HOMEDIR=/home/rbarret8/pymc3_qspr/gibbs/python/build
NUM_MOTIFS=$SLURM_ARRAY_TASK_ID
cd $HOMEDIR
python3  gibbs_cpp.py $DATADIR/APD_GPOS_txt.txt $NUM_MOTIFS 3000 0.005 1
