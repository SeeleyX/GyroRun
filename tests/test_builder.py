# tests/test_builder.py
import os
from pathlib import Path

import numpy as np
from pyrokinetics import Pyro, template_dir
from gyrorun.builder import create_scan, execute_scan
from gyrorun.run_scan import main

BASE = str(template_dir / "input.cgyro")
SLURM = {"partition": "p", "account": "a", "time_limit": "00:10:00", "run_command": "srun code"}


def test_scan_is_outer_product(tmp_path):
    scan = {
        "scan": [
            {"parameter": "ky", "values": [0.1, 0.2]},
            {"parameter": "shat", "attr": "local_geometry", "location": ["shat"], "values": [1.0, 2.0]},
        ]
    }
    jobs = execute_scan(BASE, scan, SLURM, output_dir_base=str(tmp_path / "s"), dry_run=True)

    assert len(jobs["run_dirs"]) == 4
    assert all(os.path.exists(os.path.join(d, "input.cgyro")) for d in jobs["run_dirs"])
    assert len((tmp_path / "s" / "runs.txt").read_text().splitlines()) == 4
    array = (tmp_path / "s" / "slurm_array.sh").read_text()
    assert "--array=0-3%" in array and "srun code" in array
    assert "-m gyrorun.save" in (tmp_path / "s" / "slurm_save.sh").read_text()


def test_cube_samples_within_range_and_seeded(tmp_path):
    cube = {
        "cube": [
            {"n_samples": 5},
            {"parameter": "ky", "values": [0.1, 1.0]},
            {"parameter": "shat", "attr": "local_geometry", "location": ["shat"], "values": [0.5, 3.0]},
            {"flags": [{"seed": 0}]},
        ]
    }
    dirs = create_scan(BASE, cube, output_dir_base=str(tmp_path / "a"))
    create_scan(BASE, cube, output_dir_base=str(tmp_path / "b"))

    assert [Path(d).name for d in dirs] == [f"sample_{i:04d}" for i in range(5)]
    for d in dirs:
        a = Pyro(gk_file=Path(d) / "input.cgyro")
        b = Pyro(gk_file=tmp_path / "b" / Path(d).name / "input.cgyro")
        ky, shat = a.numerics.ky.m, a.local_geometry.shat.m
        assert 0.1 <= ky <= 1.0 and 0.5 <= shat <= 3.0
        assert np.isclose(ky, b.numerics.ky.m) and np.isclose(shat, b.local_geometry.shat.m)


def test_example_pyrocube_builds(tmp_path):
    example = Path(__file__).parents[1] / "examples" / "pyrocube.yaml"
    config = tmp_path / "pyrocube.yaml"
    text = example.read_text().replace("../tests/data", str(example.parents[1] / "tests" / "data"))
    config.write_text(text.replace("../scans", "scans"))

    main([str(config)])

    scan_dir = tmp_path / "scans" / "pyrocube_example" / "scan_0"
    assert len((scan_dir / "runs.txt").read_text().splitlines()) == 20
    assert (scan_dir / "pyroscan.json").exists()
