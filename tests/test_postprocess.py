# tests/test_postprocess.py
import pytest
import numpy as np
from postprocess import analyze_single_run, collect_scan_data

def test_analyze_single_run_matrix_solver(tmp_path):
    # Setup dummy directory
    scan_dir = tmp_path / "scan_test"
    scan_dir.mkdir()
    
    # 1. Write parameters file with 1 active + 4 trace species
    param_content = """
    &species
      name = 'ion', passive = F, omn = 2.0, omt = 2.0, uprim = 0.0, dens = 1.0
    /
    &species
      name = 'trace_1', passive = T, omn = 1.0, omt = 0.0, uprim = 0.0, dens = 1.0
    /
    &species
      name = 'trace_2', passive = T, omn = 0.0, omt = 1.0, uprim = 0.0, dens = 1.0
    /
    &species
      name = 'trace_3', passive = T, omn = 0.0, omt = 0.0, uprim = 1.0, dens = 1.0
    /
    &species
      name = 'trace_4', passive = T, omn = 1.0, omt = 1.0, uprim = 1.0, dens = 1.0
    /
    """
    (scan_dir / "parameters").write_text(param_content)
    
    # 2. Write omega.dat (ky=0.3, gamma=0.15, omega=0.40)
    (scan_dir / "omega.dat").write_text("0.30  0.15  0.40\n")
    
    # 3. Write nrg.dat with fluxes calculated from:
    # Gamma = D*omn + (D*CT)*omt + (D*Cu)*uprim + (D*Cp)
    # Using D=2.0, CT=0.5 (D*CT=1.0), Cu=-0.2 (D*Cu=-0.4), Cp=0.1 (D*Cp=0.2)
    # Trace 1: 2.0(1) + 1.0(0) - 0.4(0) + 0.2 = 2.2
    # Trace 2: 2.0(0) + 1.0(1) - 0.4(0) + 0.2 = 1.2
    # Trace 3: 2.0(0) + 1.0(0) - 0.4(1) + 0.2 = -0.2
    # Trace 4: 2.0(1) + 1.0(1) - 0.4(1) + 0.2 = 2.8
    nrg_content = (
        "1.0 0 0 0 5.0 0.0\n"   # Species 1 (Active ion)
        "1.0 0 0 0 2.2 0.0\n"   # Species 2 (Trace 1)
        "1.0 0 0 0 1.2 0.0\n"   # Species 3 (Trace 2)
        "1.0 0 0 0 -0.2 0.0\n"  # Species 4 (Trace 3)
        "1.0 0 0 0 2.8 0.0\n"   # Species 5 (Trace 4)
    )
    (scan_dir / "nrg.dat").write_text(nrg_content)
    
    # Execute single run analysis
    results = analyze_single_run(str(scan_dir))
    
    # Verify eigenvalues
    assert results['ky'] == pytest.approx(0.30)
    assert results['gamma'] == pytest.approx(0.15)
    assert results['omega'] == pytest.approx(0.40)
    
    # Verify transport coefficients extracted from system solution
    assert results['D'] == pytest.approx(2.0)
    assert results['C_T'] == pytest.approx(0.5)
    assert results['C_u'] == pytest.approx(-0.2)
    assert results['C_p'] == pytest.approx(0.1)


def test_collect_scan_data(tmp_path):
    # Verify multi-folder aggregation into a pandas DataFrame
    dir1 = tmp_path / "scan_1"
    dir1.mkdir()
    (dir1 / "parameters").write_text("&geometry\n q0 = 1.5\n/\n")
    
    def mock_analysis(scan_dir):
        return {'D': 2.0, 'gamma': 0.1}
        
    df = collect_scan_data([str(dir1)], mock_analysis)
    
    assert len(df) == 1
    assert df.loc[0, 'folder'] == "scan_1"
    assert df.loc[0, 'D'] == 2.0
    assert df.loc[0, 'gamma'] == 0.1