# tests/test_builder.py
import json
import re
import os
from pathlib import Path

import numpy as np
import pytest
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

    scan_dir = tmp_path / "scans" / "CGYRO" / "Runs" / "pyrocube_example" / "CGYRO_test" / "ky_shat_n20"
    assert len((scan_dir / "runs.txt").read_text().splitlines()) == 20
    assert (scan_dir / "pyroscan.json").exists()
    manifest = json.loads((scan_dir / "manifest.json").read_text())
    assert manifest["dry_run"] is True and manifest["units"] == "tglf"


def bases_from_one_pyro(tmp_path):
    """One pyro template written as the three codes a YAML might target."""
    pyro = Pyro(gk_file=BASE)
    for code, name in [("GFTM", "input.gftm"), ("CGYRO", "input.cgyro"), ("GS2", "input.gs2")]:
        pyro.write_gk_file(tmp_path / code / name, gk_code=code)
        yield tmp_path / code / name


def test_units_give_same_physical_ky_in_every_code(tmp_path):
    scan = {"scan": [{"parameter": "ky", "values": [0.1, 0.3]}, {"units": "tglf"}]}
    for base in list(bases_from_one_pyro(tmp_path)):
        dirs = create_scan(str(base), scan, output_dir_base=str(base.parent / "scan"))
        deck = {"input.gftm": "input.gftm", "input.cgyro": "input.cgyro", "input.gs2": "input.in"}[base.name]
        decks = [Path(d) / deck for d in dirs]  # fails on pyro's old input.GFTM
        kys = sorted((p := Pyro(gk_file=f)).numerics.ky.to(p.norms.tglf).m for f in decks)
        assert np.allclose(kys, [0.1, 0.3]), (base.name, kys)


def test_cube_with_units_samples_within_converted_range(tmp_path):
    cube = {
        "cube": [
            {"n_samples": 5},
            {"parameter": "ky", "values": [0.1, 1.0]},
            {"parameter": "shat", "attr": "local_geometry", "location": ["shat"], "values": [0.5, 3.0]},
            {"flags": [{"seed": 0}]},
            {"units": "gs2"},
        ]
    }
    for d in create_scan(BASE, cube, output_dir_base=str(tmp_path / "c")):
        p = Pyro(gk_file=Path(d) / "input.cgyro")
        assert 0.1 <= p.numerics.ky.to(p.norms.gs2).m <= 1.0
        assert 0.5 <= p.local_geometry.shat.m <= 3.0


def test_omitted_units_take_the_base_inputs_own(tmp_path):
    # today's behaviour: plain floats get the base deck's units, so the same
    # YAML means a different physical ky per code
    base = [b for b in bases_from_one_pyro(tmp_path) if b.name == "input.gs2"][0]
    scan = {"scan": [{"parameter": "ky", "values": [0.1]}]}
    (d,) = create_scan(str(base), scan, output_dir_base=str(tmp_path / "s"))
    p = Pyro(gk_file=Path(d) / "input.in")
    assert np.isclose(p.numerics.ky.to(p.norms.gs2).m, 0.1)
    assert not np.isclose(p.numerics.ky.to(p.norms.tglf).m, 0.1)


def test_unknown_convention_fails_loudly(tmp_path):
    scan = {"scan": [{"parameter": "ky", "values": [0.1]}, {"units": "tglff"}]}
    with pytest.raises(ValueError, match="tglff"):
        create_scan(BASE, scan, output_dir_base=str(tmp_path / "s"))


def test_qos_line_only_when_given(tmp_path):
    scan = {"scan": [{"parameter": "ky", "values": [0.1]}]}
    execute_scan(BASE, scan, {**SLURM, "qos": "normal"}, output_dir_base=str(tmp_path / "q"))
    execute_scan(BASE, scan, SLURM, output_dir_base=str(tmp_path / "n"))
    for script in ["slurm_array.sh", "slurm_save.sh"]:
        assert "#SBATCH --qos=normal" in (tmp_path / "q" / script).read_text()
        assert "--qos" not in (tmp_path / "n" / script).read_text()


def gs2_template_tree(tmp_path, n=3):
    """A small per-case GS2 database: n runs, ky varied."""
    pyro = Pyro(gk_file=BASE)
    pyro.convert_gk_code("GS2")
    root = tmp_path / "db"
    for i in range(n):
        pyro.write_gk_file(root / f"case_{i}" / "gs2.in", gk_code="GS2")
    return root


def deck_text(d):
    return (Path(d) / "input.gftm").read_text()


def test_gk_code_and_code_flags_written_to_deck(tmp_path):
    gs2 = tmp_path / "gs2" / "gs2.in"
    Pyro(gk_file=BASE).write_gk_file(gs2, gk_code="GS2")
    scan = {"scan": [{"parameter": "ky", "values": [0.1, 0.3]}, {"units": "tglf"},
                     {"gk_code": "GFTM"}, {"code_flags": {"WIDTH": 0.6, "FILTER": 0.5}}]}
    dirs = create_scan(str(gs2), scan, output_dir_base=str(tmp_path / "s"))
    assert len(dirs) == 2
    for d in dirs:
        text = deck_text(d)
        assert re.search(r"^WIDTH\s*=\s*0\.6", text, re.M)
        assert re.search(r"^FILTER\s*=\s*0\.5", text, re.M)


def test_from_directory_cube_converts_and_sets_flags(tmp_path):
    root = gs2_template_tree(tmp_path)
    cube = {"cube": [
        {"from_directory": {"root": str(root), "pattern": "case_*", "params": ["ky"], "gk_code": "GS2"}},
        {"gk_code": "GFTM"}, {"code_flags": {"WIDTH": 0.6, "FILTER": 0.5}},
    ]}
    dirs = create_scan(None, cube, output_dir_base=str(tmp_path / "out"))
    assert sorted(Path(d).name for d in dirs) == ["case_0", "case_1", "case_2"]
    assert not list((root).rglob("input.gftm")), "source database must be untouched"
    for d in dirs:
        assert re.search(r"^FILTER\s*=\s*0\.5", deck_text(d), re.M)


def test_leaf_follows_converted_code_and_manifest_records_flags(tmp_path):
    gs2 = tmp_path / "in" / "gs2.in"
    Pyro(gk_file=BASE).write_gk_file(gs2, gk_code="GS2")
    config = tmp_path / "c.yaml"
    config.write_text(f"""
paths: {{base_input: {gs2}, output_dir_base: {tmp_path / 'o'}}}
slurm: {{partition: p, account: a, time_limit: "00:10:00", run_command: srun x}}
run: {{dry_run: true, scan_name: proj, case: M1, gk_code: GFTM}}
scans:
  - name: leaf
    scan:
      - parameter: ky
        values: [0.1, 0.3]
      - units: tglf
      - code_flags: {{WIDTH: 0.6, FILTER: 0.5}}
""")
    main([str(config)])
    leaf = tmp_path / "o" / "GFTM" / "Runs" / "proj" / "M1" / "leaf"
    manifest = json.loads((leaf / "manifest.json").read_text())
    assert manifest["code_flags"] == {"WIDTH": 0.6, "FILTER": 0.5} and manifest["gk_code"] == "GFTM"
    assert len(list(leaf.glob("*/input.gftm"))) == 2
