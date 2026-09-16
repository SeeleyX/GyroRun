# transport_sensitivity/slurm.py
import os
import time
import subprocess
import re
import random
from string import Template

# A basic template. In the future, this could be read from a file.
DEFAULT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=01:00:00

# Load necessary modules (adjust for your cluster)
# module load intel/2023 openmpi/4.1

# Run the executable
# mpirun -np 32 ./gene parameters
echo "Running GENE in $PWD"
"""

def generate_sbatch_script(template_path, output_path, context):
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Slurm template not found at {template_path}")

    with open(template_path, "r") as f:
        template_str = f.read()

    script_content = template_str.format(**context)

    with open(output_path, "w") as f:
        f.write(script_content)

    return output_path
    
def submit_job(script_path, scan_dir, dry_run=False):
    if dry_run:
        fake_id = random.randint(100000, 999999)
        print(f"[DRY RUN] Would submit: sbatch {script_path} {scan_dir}")
        return fake_id

    try:
        result = subprocess.run(
            ["sbatch", os.path.basename(script_path), os.path.abspath(scan_dir)],
            cwd=scan_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        # sbatch output usually looks like: "Submitted batch job 123456"
        output = result.stdout.strip()
        print(output)
        
        # Extract the job ID using regex
        match = re.search(r"Submitted batch job (\d+)", output)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Could not parse job ID from output: {output}")
            
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job in {scan_dir}. Error: {e.stderr}")
        raise

def get_job_statuses(job_ids, dry_run=False):
    """
    Queries Slurm for the status of a list of job IDs.
    Returns a dict: {job_id: 'COMPLETED'|'RUNNING'|'FAILED'|etc.}
    """
    if dry_run:
        # In dry run mode, pretend all jobs instantly completed
        return {job_id: "COMPLETED" for job_id in job_ids}

    # Format job IDs for sacct: "1234,1235,1236"
    job_str = ",".join(str(j) for j in job_ids)
    
    try:
        # sacct -j <IDs> --format=JobID,State -n -P (parsable, no header)
        result = subprocess.run(
            ["sacct", "-j", job_str, "--format=JobID,State", "-n", "-P"],
            capture_output=True,
            text=True,
            check=True
        )
        
        statuses = {}
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            job_id_raw, state = parts[0], parts[1]
            
            # Filter out sub-step jobs like "12345.batch" or "12345.0"
            if "." not in job_id_raw:
                statuses[int(job_id_raw)] = state
                
        return statuses

    except subprocess.CalledProcessError as e:
        print(f"Error checking job status: {e.stderr}")
        return {}

def wait_for_jobs(job_dirs, poll_interval=30, dry_run=False):
    """
    Blocks execution until all tracked Slurm jobs are finished.
    job_dirs: dict mapping {job_id: directory_path}
    """
    print(f"\nMonitoring {len(job_dirs)} jobs...")
    
    if dry_run:
        print("[DRY RUN] Skipping wait loop. All jobs marked COMPLETED.")
        return True

    pending_jobs = list(job_dirs.keys())
    
    while pending_jobs:
        statuses = get_job_statuses(pending_jobs, dry_run=dry_run)
        
        still_running = []
        for job_id in pending_jobs:
            state = statuses.get(job_id, "UNKNOWN")
            
            if state in ["COMPLETED"]:
                print(f"  Job {job_id} ({job_dirs[job_id]}) finished successfully.")
            elif state in ["FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]:
                print(f"  WARNING: Job {job_id} failed with state: {state}")
            else:
                # Job is still PENDING or RUNNING
                still_running.append(job_id)
                
        pending_jobs = still_running
        
        if pending_jobs:
            print(f"  {len(pending_jobs)} jobs remaining. Sleeping for {poll_interval}s...")
            time.sleep(poll_interval)
            
    print("All jobs finished!\n")
    return True