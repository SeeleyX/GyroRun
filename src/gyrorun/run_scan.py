# run_scan.py
import os
import yaml
from .builder import execute_scan
from .slurm import wait_for_jobs

# repo root, two levels above src/gyrorun/
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

def main():
    with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
        config = yaml.safe_load(f)

    paths = {
        k: v if os.path.isabs(v) else os.path.join(PROJECT_ROOT, v)
        for k, v in config["paths"].items()
    }

    run_name = config["run"]["name"]
    paths["output_dir_base"] = os.path.join(paths["output_dir_base"], run_name)

    slurm_cfg = config["slurm"]
    dry_run = config["run"]["dry_run"]

    all_scan_dirs = []
    for scan_cfg in config["scans"]:
        active_jobs = execute_scan(
            base_filepath=paths["base_parameters"],
            scan_config=scan_cfg,
            output_dir_base=paths["output_dir_base"],
            template_path=paths["slurm_template"],
            gene_executable=paths["gene_executable"],
            poll_interval=slurm_cfg["poll_interval"],
            slurm_kwargs={
                "gene_executable": paths["gene_executable"],
                "partition": slurm_cfg["partition"],
                "account": slurm_cfg["account"],
                "time_limit": slurm_cfg["time_limit"],
            },
            dry_run=dry_run,
            wait=False   
        )
        all_scan_dirs.extend(active_jobs.values())

    return all_scan_dirs


if __name__ == "__main__":
    main()
