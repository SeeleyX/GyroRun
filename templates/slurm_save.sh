#!/bin/bash
#SBATCH --job-name={job_name}_save
#SBATCH --output=slurm-save-%j.out
#SBATCH --error=slurm-save-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time={save_time_limit}
#SBATCH --partition={partition}
#SBATCH --account={account}

{python} -m gyrorun.save {scan_dir}
