# tests/test_save.py
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
from gyrorun.save import save

# pyro's public CGYRO_linear_scan output (templates/outputs is not in its wheel)
DATA = Path(__file__).parent / "data" / "CGYRO_linear_scan"


def test_save_writes_netcdf(tmp_path):
    scan_dir = shutil.copytree(DATA, tmp_path / "scan")
    save(scan_dir)
    with xr.open_dataset(scan_dir / "pyroscan.nc") as ds:
        assert np.isfinite(ds["growth_rate"]).all()
