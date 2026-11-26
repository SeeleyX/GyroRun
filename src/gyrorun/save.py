"""Write a scan's output to pyroscan.nc: python -m gyrorun.save <scan_dir> [--tearing] [--store <dir>]"""
import shutil
import json
import os
import sys
from pathlib import Path
from pyrokinetics import PyroScan, PyroHypercube


SIDECARS = ("pyroscan.json", "pyroscan_base.input", "pyroscan_norms.json")


def save(scan_dir, tearing=False, store=None):
    """store: write pyroscan.nc and the sidecars here instead of scan_dir."""
    json_file = Path(scan_dir) / "pyroscan.json"
    cls = PyroHypercube if "sample_names" in json.loads(json_file.read_text()) else PyroScan
    scan = cls(pyroscan_json=json_file, load_base_pyro=True)
    scan.load_gk_output(load_tearing_parameter=tearing)
    out = Path(store) if store else Path(scan_dir)
    out.mkdir(parents=True, exist_ok=True)
    scan.gk_output.to_netcdf(out / "pyroscan.nc")
    if store:
        for name in SIDECARS:
            if (Path(scan_dir) / name).exists():
                shutil.copy2(Path(scan_dir) / name, out / name)


if __name__ == "__main__":
    args = sys.argv[2:]
    store = os.path.expandvars(args[args.index("--store") + 1]) if "--store" in args else None
    save(sys.argv[1], tearing="--tearing" in args, store=store)
