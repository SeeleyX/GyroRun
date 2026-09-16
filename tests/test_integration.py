# tests/test_integration.py
import os
import pytest
from builder import execute_scan

@pytest.fixture
def base_nml_file(tmp_path):
    p = tmp_path / "base_parameters"
    content = """
&parallelization
  n_procs_sim = 64
  n_procs_s = 1
  n_procs_v = 1
  n_procs_w = 8
  n_procs_x = 1
  n_procs_y = 1
  n_procs_z = 8
/
&species name = 'ion' /
&geometry
  beta = 0.002
  betaprime = -0.05
  q0 = 1.5
/
&parameters
  kymin = 0.3
/
"""
    p.write_text(content)
    return str(p)

def test_full_pipeline_dry_run(base_nml_file, tmp_path):
    scan_dir = str(tmp_path / "scans")
    scan_config = {'group': 'geometry', 'param': 'q0', 'values': [1.5, 2.0]}
    
    active_jobs = execute_scan(
        base_nml_file, scan_config,
        output_dir_base=scan_dir,
        gene_executable="/fake/path/gene_uprim",
        poll_interval=1,
        dry_run=True
    )    
    
    assert len(active_jobs) == 2
    for job_id, folder in active_jobs.items():
        assert isinstance(job_id, int)
        assert os.path.exists(os.path.join(folder, "run_gene.sh"))
        assert os.path.exists(os.path.join(folder, "parameters"))