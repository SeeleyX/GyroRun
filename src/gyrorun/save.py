"""Write a scan's output to pyroscan.nc: python -m gyrorun.save <scan_dir>"""
import json
import sys
from pathlib import Path
from pyrokinetics import PyroScan, PyroHypercube


def save(scan_dir):
    json_file = Path(scan_dir) / "pyroscan.json"
    cls = PyroHypercube if "sample_names" in json.loads(json_file.read_text()) else PyroScan
    scan = cls(pyroscan_json=json_file, load_base_pyro=True)
    scan.load_gk_output()
    scan.gk_output.to_netcdf(Path(scan_dir) / "pyroscan.nc")


if __name__ == "__main__":
    save(sys.argv[1])
