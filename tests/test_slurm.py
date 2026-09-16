# tests/test_slurm.py
import os
import subprocess
import pytest
from unittest.mock import patch
from gyrorun.slurm import generate_sbatch_script, submit_job

def test_generate_sbatch_from_template(tmp_path):
    template = tmp_path / "slurm_template.sh"
    template.write_text("#!/bin/bash\n#SBATCH -n {ntasks}\ncd {run_dir}\n")

    out_script = tmp_path / "submit.sh"
    context = {"ntasks": 32, "run_dir": "/tmp/test_dir"}

    generate_sbatch_script(str(template), str(out_script), context)

    content = out_script.read_text()
    assert "#SBATCH -n 32" in content

def test_submit_job_dry_run(tmp_path):
    job_id = submit_job("fake_script.sh", str(tmp_path), dry_run=True)
    assert isinstance(job_id, int)
    assert job_id > 0


def test_submit_job_dry_run(tmp_path):
    job_id = submit_job("fake_script.sh", str(tmp_path), dry_run=True)
    assert isinstance(job_id, int)
    assert job_id > 0

# The @patch decorator intercepts the subprocess.run call so it doesn't actually try to run sbatch!
@patch('gyrorun.slurm.subprocess.run')
def test_submit_job_mocked(mock_run, tmp_path):
    # 1. Setup the fake response that sbatch *would* give us
    mock_run.return_value.stdout = "Submitted batch job 123456\n"
    mock_run.return_value.returncode = 0
    
    # 2. Call our function
    job_id = submit_job("run_gene.sh", str(tmp_path), dry_run=False)
    
    # 3. Assert our code successfully extracted the Job ID from the fake response
    assert job_id == 123456
    
    # 4. Verify our code called subprocess.run with the correct arguments
    mock_run.assert_called_once_with(
        ["sbatch", "run_gene.sh", str(tmp_path)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=True
    )