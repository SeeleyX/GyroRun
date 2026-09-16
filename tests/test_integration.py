# tests/test_integration.py
import os
from pathlib import Path

import pytest
from gyrorun.builder import execute_scan

@pytest.fixture
def base_nml_file(tmp_path):
    p = tmp_path / "base_parameters"
    template = Path(__file__).parents[1] / "templates" / "parameters_template"
    p.write_text(template.read_text())
    return str(p)

def test_full_pipeline_dry_run(base_nml_file, tmp_path):
    scan_dir = str(tmp_path / "scans")
    scan_config = {
        'scan': [
            {
                'parameter': 'q0',
                'attr': 'local_geometry',
                'location': ['q'],
                'values': [1.5, 2.0],
            },
            {'flags': ['enforce_consistent_beta_prime']},
        ],
    }
    
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
        assert os.path.exists(os.path.join(folder, "input.gene"))
