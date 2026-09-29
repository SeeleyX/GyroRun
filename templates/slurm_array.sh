#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=0-$last_index%$max_parallel
#SBATCH --nodes=$nodes
#SBATCH --ntasks=$ntasks
#SBATCH --time=$time_limit
#SBATCH --partition=$partition
#SBATCH --account=$account
$qos_line

# task i runs in the directory on line i+1 of runs.txt
cd "$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $runs_file)" || exit 1

$setup
$run_command
