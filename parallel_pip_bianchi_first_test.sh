#!/bin/bash -l

# Submit this script as: "./prepare-env.sh" instead of "sbatch prepare-env.sh"

# Prepare user env needed for Slurm batch job
# such as module load, setup runtime environment variables, or copy input files, etc.
# Basically, these are the commands you usually run ahead of the srun command 

source /project/projectdirs/desi/software/desi_environment.sh 19.2
module unload fiberassign
module swap PrgEnv-intel PrgEnv-gnu
export CONDA_ENVS_PATH=$SCRATCH/desi/conda
source activate desienv
export PYTHONPATH=$SCRATCH/desi/conda/desienv/lib/python3.6/site-packages:$PYTHONPATH

# Generate the Slurm batch script below with the here document, 
# then when sbatch the script later, the user env set up above will run on the login node
# instead of on a head compute node (if included in the Slurm batch script),
# and inherited into the batch job.

cat << EOF > prepare-env.sl 
#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time-min=01:00:00
#SBATCH --time=02:00:00
#SBATCH --output='../output/parallel_pip_bianchi_first_test.out'
#SBATCH --error='../output/parallel_pip_bianchi_first_test.err'
#SBATCH --qos=regular
#SBATCH --account=desi
#SBATCH --job-name=pip_lrg

srun -c 64 python ./parallel_pip_bianchi_first_test.py

# Other commands needed after srun, such as copy your output filies,
# should still be incldued in the Slurm script.
EOF

# Now submit the batch job
sbatch prepare-env.sl
