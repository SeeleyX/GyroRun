# tests/test_builder.py
import os
import pytest
import f90nml
from builder import create_1d_scan

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

# 2. Test the simple 1D scan
def test_simple_1d_scan(base_nml_file, tmp_path):
    scan_dir = str(tmp_path / "scans")
    simple_scan = {
        'group': 'geometry',
        'param': 'q0',
        'values': [2.0, 3.0]
    }
    
    # Run the function
    generated_dirs = create_1d_scan(base_nml_file, simple_scan, output_dir_base=scan_dir)
    
    # Assert the correct number of folders were created
    assert len(generated_dirs) == 2
    
    # Assert the files exist and the values were actually updated
    folder_2 = os.path.join(scan_dir, "scan_q0_2.0")
    assert os.path.exists(os.path.join(folder_2, "parameters"))
    
    nml = f90nml.read(os.path.join(folder_2, "parameters"))
    assert nml['geometry']['q0'] == 2.0
    assert nml['parameters']['kymin'] == 0.3  # Ensure untouched variables remain

# 3. Test the self-consistent function logic
def test_consistent_beta_scan(base_nml_file, tmp_path):
    scan_dir = str(tmp_path / "scans")
    
    def scale_beta(new_beta):
        def updater(base_nml):
            base_beta = base_nml['geometry']['beta']
            base_betaprime = base_nml['geometry']['betaprime']
            scaling_factor = new_beta / base_beta
            new_betaprime = base_betaprime * scaling_factor
            return [
                ('geometry', 'beta', new_beta),
                ('geometry', 'betaprime', new_betaprime)
            ]
        return updater

    consistent_scan = {
        'group': 'geometry',
        'param': 'beta_scaled',
        'values': [scale_beta(0.004)],
        'labels': ["0.004"]
    }
    
    create_1d_scan(base_nml_file, consistent_scan, output_dir_base=scan_dir)
    
    # Verify both beta and betaprime scaled correctly
    output_nml = f90nml.read(os.path.join(scan_dir, "scan_beta_scaled_0.004", "parameters"))
    assert output_nml['geometry']['beta'] == 0.004
    assert output_nml['geometry']['betaprime'] == -0.10  # -0.05 * (0.004 / 0.002)