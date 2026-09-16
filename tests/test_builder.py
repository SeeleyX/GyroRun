# tests/test_builder.py
import os
import pytest
import f90nml
from builder import create_scan


# 1. Setup a fixture (creates a temporary base parameter file for every test)
@pytest.fixture
def base_nml_file(tmp_path):
    filepath = tmp_path / "base_parameters"
    content = """&parameters
  kymin = 0.3
/
&geometry
  q0 = 1.5
  beta = 0.002
  betaprime = -0.05
/
"""
    filepath.write_text(content)
    return str(filepath)


def test_create_scan_generates_real_pyroscan_in_temporary_directory(
    base_nml_file, tmp_path
):

    scan_dir = str(tmp_path / "scans")
    simple_scan = {
        "scan": [
            {
                "parameter": "q0",
                "attr": "local_geometry",
                "location": ["q"],
                "values": [1.35, 1.5],
            },
            {
                "flags": ["enforce_consistent_beta_prime"],
            },
        ]
    }

    generated_dirs = create_scan(base_nml_file, simple_scan, output_dir_base=scan_dir)

    # Assert the correct number of folders were created
    assert len(generated_dirs) == 2

    # Assert the files exist and the values were actually updated
    folder_2 = os.path.join(scan_dir, "scan_q0_2.0")
    assert os.path.exists(os.path.join(folder_2, "parameters"))

    nml = f90nml.read(os.path.join(folder_2, "parameters"))
    assert nml["geometry"]["q0"] == 2.0
    assert nml["parameters"]["kymin"] == 0.3  # Ensure untouched variables remain
