# run_scan.py
import os
import sys
import yaml
from .builder import execute_scan


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
    run_dir = os.path.join(paths["output_dir_base"], config["run"]["name"])

    for i, scan_cfg in enumerate(config["scans"]):
        jobs = execute_scan(
            base_filepath=paths["base_input"],
            scan_config=scan_cfg,
            slurm_config=config["slurm"],
            output_dir_base=os.path.join(run_dir, f"scan_{i}"),
            array_template=paths.get("slurm_template"),
            save_template=paths.get("save_template"),
            dry_run=config["run"]["dry_run"],
        )
        print(f"scan_{i}: {len(jobs['run_dirs'])} runs, array job {jobs['array']}, save job {jobs['save']}")


if __name__ == "__main__":
    main()
