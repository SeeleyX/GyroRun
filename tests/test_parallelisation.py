# tests/test_validation.py
import pytest
import f90nml
from builder import validate_parallelization

VALID_NML = """
&parallelization
  n_procs_sim = 64
  n_procs_s = 2
  n_procs_v = 1
  n_procs_w = 8
  n_procs_x = 1
  n_procs_y = 1
  n_procs_z = 4
/
&species name = 'ion', passive = F /
&species name = 'electron', passive = F /
&species name = 'trace1', passive = T /
&species name = 'trace2', passive = T /
"""

def test_valid_parallelization():
    nml = f90nml.reads(VALID_NML)
    # 4 species total, n_procs_s = 2 (divisible), n_procs_sim = 64 matches Slurm 64
    assert validate_parallelization(nml, slurm_ntasks=64) is True

def test_invalid_species_factor():
    # 3 species total is NOT divisible by n_procs_s = 2
    invalid_nml = """
    &parallelization
      n_procs_sim = 64
      n_procs_s = 2
      n_procs_v = 1
      n_procs_w = 8
      n_procs_x = 1
      n_procs_y = 1
      n_procs_z = 4
    /
    &species name = 'ion' /
    &species name = 'electron' /
    &species name = 'trace1' /
    """
    nml = f90nml.reads(invalid_nml)
    with pytest.raises(ValueError, match="must be divisible by n_procs_s"):
        validate_parallelization(nml, slurm_ntasks=64)

def test_slurm_procs_mismatch():
    nml = f90nml.reads(VALID_NML)
    # n_procs_sim is 64, but Slurm script requests 32
    with pytest.raises(ValueError, match="does not match Slurm requested ntasks"):
        validate_parallelization(nml, slurm_ntasks=32)