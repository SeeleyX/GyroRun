# run_scan.py
import os
import sys
import yaml
from pyrokinetics import Pyro
from .builder import execute_scan
from .layout import compose_leaf, write_manifest


def main(argv=None):
    """gyrorun config.yaml: build and submit every scan in the config."""
    config_path = os.path.abspath((argv or sys.argv[1:] or ["config.yaml"])[0])
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # relative paths are relative to the config file; env vars are expanded
    config_dir = os.path.dirname(config_path)
    paths = {
        k: os.path.join(config_dir, os.path.expanduser(os.path.expandvars(v)))
        for k, v in config["paths"].items()
    }
    run = config["run"]
    override = paths.get("output_dir_base")
    input_root = os.environ.get("GYRO_DATA_INPUT")
    code = Pyro(gk_file=paths["base_input"]).gk_code
    if input_root and not os.path.abspath(paths["base_input"]).startswith(
        os.path.join(input_root, code.upper(), "Templates") + os.sep
    ):
        print(f"WARNING: base_input is not under $GYRO_DATA_INPUT/{code.upper()}/Templates/")

    # compose and check every leaf before writing any
    leaves = [compose_leaf(code, run, s["name"], override) for s in config["scans"]]
    if len(set(leaves)) < len(leaves):
        raise SystemExit("two scans share a leaf name")

    for scan_cfg, leaf in zip(config["scans"], leaves):
        jobs = execute_scan(
            base_filepath=paths["base_input"],
            scan_config=scan_cfg,
            slurm_config=config["slurm"],
            output_dir_base=str(leaf),
            array_template=paths.get("slurm_template"),
            save_template=paths.get("save_template"),
            dry_run=run["dry_run"],
        )
        write_manifest(leaf, config, scan_cfg, {**jobs, "dry_run": run["dry_run"]},
                       paths.get("executable"))
        print(f"{leaf}: {len(jobs['run_dirs'])} runs, array job {jobs['array']}, save job {jobs['save']}")
    print("\nleaves for docs/gyro_data_runs.md:")
    print("\n".join(f"  {leaf}" for leaf in leaves))

if __name__ == "__main__":
    main()
