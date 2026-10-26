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

# task i runs the directories on lines i*k+1 .. (i+1)*k of runs.txt, k = runs_per_task
k=$runs_per_task
runs=$(sed -n "$((SLURM_ARRAY_TASK_ID*k+1)),$(((SLURM_ARRAY_TASK_ID+1)*k))p" $runs_file)
cd "$(head -n1 <<< "$runs")" || exit 1

$setup
status=0
for run_dir in $runs; do
    cd "$run_dir" || exit 1
    $run_command || status=1
done
exit $status
