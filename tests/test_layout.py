import json
import pytest
from pyrokinetics import template_dir
from gyrorun.layout import compose_leaf
from gyrorun.run_scan import main

RUN = {"scan_name": "proj", "case": "M1"}


def test_composed_path_per_code(tmp_path):
    assert compose_leaf("cgyro", RUN, "a/b", tmp_path) == tmp_path / "CGYRO/Runs/proj/M1/a/b"
    assert compose_leaf("TGLF", RUN, "x", tmp_path) == tmp_path / "TGLF/Runs/proj/M1/x"


def test_env_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GYRO_DATA_OUTPUT", str(tmp_path))
    assert compose_leaf("GS2", RUN, "x") == tmp_path / "GS2/Runs/proj/M1/x"


def test_refuses_nonempty_leaf_and_bad_segment(tmp_path):
    leaf = tmp_path / "TGLF/Runs/proj/M1/x"
    leaf.mkdir(parents=True)
    compose_leaf("TGLF", RUN, "x", tmp_path)  # empty is fine
    (leaf / "f").write_text("")
    with pytest.raises(SystemExit, match="non-empty"):
        compose_leaf("TGLF", RUN, "x", tmp_path)
    for bad in ["a b", "../x", "a//b", "a/$x"]:
        with pytest.raises(SystemExit, match="invalid"):
            compose_leaf("TGLF", RUN, bad, tmp_path)


def test_same_param_two_scans_distinct_dirs_and_one_array_script(tmp_path):
    cfg = tmp_path / "c.yaml"
    scan = lambda n: f"""  - name: {n}
    scan:
      - parameter: ky
        values: [0.1, 0.2, 0.3]
"""
    cfg.write_text(f"""paths:
  base_input: {template_dir / 'input.tglf'}
  output_dir_base: out
slurm: {{partition: p, account: a, time_limit: "00:10:00", run_command: 'echo ${{SLURM_JOB_ID}}'}}
run: {{dry_run: true, scan_name: proj, case: M1}}
scans:
{scan('a')}{scan('b')}""")
    main([str(cfg)])
    base = tmp_path / "out/TGLF/Runs/proj/M1"
    for n in "ab":
        assert len(list(base.glob(f"{n}/slurm_array.sh"))) == 1
        assert "${SLURM_JOB_ID}" in (base / n / "slurm_array.sh").read_text()
        assert json.loads((base / n / "manifest.json").read_text())["array_job"]
    with pytest.raises(SystemExit, match="non-empty"):
        main([str(cfg)])
    assert "sacct" not in "".join(p.read_text() for p in base.rglob("*.sh"))


def test_dry_run_without_output_dir_base_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("GYRO_DATA_OUTPUT", str(tmp_path / "real"))
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""paths: {{base_input: {template_dir / 'input.tglf'}}}
slurm: {{partition: p, account: a, time_limit: "00:10:00", run_command: x}}
run: {{dry_run: true, scan_name: proj, case: M1}}
scans: [{{name: a, scan: [{{parameter: ky, values: [0.1]}}]}}]
""")
    with pytest.raises(SystemExit, match="output_dir_base"):
        main([str(cfg)])
    assert not (tmp_path / "real").exists()


def test_store_dir_mirrors_leaf_in_save_job(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""paths:
  base_input: {template_dir / 'input.tglf'}
  output_dir_base: out
  store_dir: {tmp_path}/store
slurm: {{partition: p, account: a, time_limit: "00:10:00", run_command: 'echo hi'}}
run: {{dry_run: true, scan_name: proj, case: M1}}
scans:
  - name: a
    scan:
      - parameter: ky
        values: [0.1, 0.2]
""")
    main([str(cfg)])
    save_sh = (tmp_path / "out/TGLF/Runs/proj/M1/a/slurm_save.sh").read_text()
    assert f"--store {tmp_path}/store/TGLF/Runs/proj/M1/a" in save_sh
