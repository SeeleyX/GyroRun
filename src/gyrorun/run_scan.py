# run_scan.py
import os
import sys
import yaml
from pyrokinetics import Pyro
from .builder import execute_scan
from .layout import compose_leaf, kind, write_manifest


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
    if run["dry_run"] and not override:
        raise SystemExit(
            "run.dry_run: true needs paths.output_dir_base, or it writes into the real "
            "$GYRO_DATA_OUTPUT tree; use e.g. $SCRATCH/gyrorun_dryrun"
        )
    input_root = os.environ.get("GYRO_DATA_INPUT")
    # the leaf's <CODE> is the code the decks are converted to: a scan's own
    # gk_code, else run.gk_code, else the base input's
    def target(scan_cfg):
        items = scan_cfg[kind(scan_cfg)]
        return next((i["gk_code"] for i in items if "gk_code" in i), run.get("gk_code"))

    base_code = Pyro(gk_file=paths["base_input"]).gk_code if "base_input" in paths else None
    if base_code and input_root and not os.path.abspath(paths["base_input"]).startswith(
        os.path.join(input_root, base_code.upper(), "Templates") + os.sep
    ):
        print(f"WARNING: base_input is not under $GYRO_DATA_INPUT/{base_code.upper()}/Templates/")

    # compose and check every leaf before writing any
    leaves = [
        compose_leaf(target(s) or base_code, run, s["name"], override) for s in config["scans"]
    ]
    if len(set(leaves)) < len(leaves):
        raise SystemExit("two scans share a leaf name")

    for scan_cfg, leaf in zip(config["scans"], leaves):
        jobs = execute_scan(
            base_filepath=paths.get("base_input"),
            scan_config=scan_cfg,
            slurm_config=config["slurm"],
            output_dir_base=str(leaf),
            array_template=paths.get("slurm_template"),
            save_template=paths.get("save_template"),
            dry_run=run["dry_run"],
            gk_code=run.get("gk_code"),
            tearing_parameter=run.get("tearing_parameter", False),
        )
        write_manifest(leaf, config, scan_cfg, {**jobs, "dry_run": run["dry_run"]},
                       paths.get("executable"))
        print(f"{leaf}: {len(jobs['run_dirs'])} runs, array job {jobs['array']}, save job {jobs['save']}")
    print("\nleaves for docs/gyro_data_runs.md:")
    print("\n".join(f"  {leaf}" for leaf in leaves))

if __name__ == "__main__":
    main()
