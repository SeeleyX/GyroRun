import os
import re
import sys
from pyrokinetics import Pyro, PyroScan, PyroHypercube
from scipy.stats import qmc
from .slurm import generate_sbatch_script, submit_job

# repo root, two levels above src/gyrorun/
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def create_scan(base_filepath, scan_config, output_dir_base="scans"):
    """
    Build a scan with PyroScan, or a cube with PyroHypercube. scan_config is
    one entry of config.yaml's 'scans' list, keyed 'scan' or 'cube':

        scan:
          - parameter: q0
            attr: local_geometry
            location: [q]
            values: [1.35, 1.5, 1.65]
          - parameter: ky
            values: [0.1, 0.3]
          - flags:
              - enforce_consistent_beta_prime

        cube:
          - n_samples: 300
          - parameter: ky
            values: [0.1, 1.0]
          - flags:
              - seed: 0

    A scan generates every combination of the values. A cube draws n_samples
    points from a Latin hypercube over the [min, max] range in each
    parameter's values. attr and location register where the quantity lives in
    the pyro object, and are only needed for parameters pyrokinetics does not
    already define. Flags apply only to the scan that declares them.
    """
    kind = "cube" if "cube" in scan_config else "scan"
    items = scan_config[kind]
    values = {item["parameter"]: item["values"] for item in items if "parameter" in item}
    flags = [flag for item in items for flag in item.get("flags", [])]

    pyro = Pyro(gk_file=base_filepath)
    if kind == "cube":
        n_samples = next(item["n_samples"] for item in items if "n_samples" in item)
        seed = next((f["seed"] for f in flags if isinstance(f, dict) and "seed" in f), None)
        lower, upper = zip(*values.values())
        # pyro has no sampler of its own; its how-to draws samples by hand too
        sample = qmc.LatinHypercube(d=len(values), rng=seed).random(n_samples)
        sample = qmc.scale(sample, lower, upper)
        values = {name: sample[:, i].tolist() for i, name in enumerate(values)}
        scan = PyroHypercube(pyro, values, base_directory=output_dir_base)
    else:
        scan = PyroScan(pyro, values, base_directory=output_dir_base)

    for item in items:
        if "parameter" in item and "attr" in item:
            scan.add_parameter_key(item["parameter"], item["attr"], item["location"])

    if "enforce_consistent_pvg" in flags and "gamma_exb" in values:
        scan.add_parameter_func("gamma_exb", Pyro.enforce_consistent_pvg, {})

    if "enforce_consistent_beta_prime" in flags and "beta" in values:

        def enforce_beta_prime(pyro):
            pyro.enforce_consistent_beta_prime()

        scan.add_parameter_func("beta", enforce_beta_prime, {})

    scan.write()

    return [str(d) for d in scan.run_directories]


def execute_scan(
    base_filepath,
    scan_config,
    slurm_config,
    output_dir_base="scans",
    array_template=None,
    save_template=None,
    dry_run=True,
):
    """
    Write the scan, then submit one Slurm array job over its run directories
    and a save job that runs after it to write pyroscan.nc. Does not wait.

    slurm_config is config.yaml's 'slurm' mapping. run_command is required;
    nodes, ntasks, max_parallel, setup and save_time_limit are optional.
    """
    if array_template is None:
        array_template = os.path.join(PROJECT_ROOT, "templates", "slurm_array.sh")
    if save_template is None:
        save_template = os.path.join(PROJECT_ROOT, "templates", "slurm_save.sh")

    run_dirs = create_scan(base_filepath, scan_config, output_dir_base=output_dir_base)
    scan_dir = os.path.abspath(output_dir_base)
    runs_file = os.path.join(scan_dir, "runs.txt")
    with open(runs_file, "w") as f:
        f.writelines(os.path.abspath(d) + "\n" for d in run_dirs)

    context = {
        "job_name": "_".join(scan_dir.split(os.sep)[-2:]),
        "nodes": 1,
        "ntasks": 1,
        "max_parallel": 50,
        "save_time_limit": "00:30:00",
        **slurm_config,
        "setup": "\n".join(slurm_config.get("setup", [])),
        "last_index": len(run_dirs) - 1,
        "runs_file": runs_file,
        "scan_dir": scan_dir,
        # the save job reads output with the interpreter, and so the pinned
        # pyrokinetics, that wrote the inputs
        "python": sys.executable,
    }

    array_script = generate_sbatch_script(
        array_template, os.path.join(scan_dir, "slurm_array.sh"), context
    )
    array_id = submit_job(array_script, scan_dir, dry_run=dry_run)

    save_script = generate_sbatch_script(
        save_template, os.path.join(scan_dir, "slurm_save.sh"), context
    )
    save_id = submit_job(
        save_script,
        scan_dir,
        dry_run=dry_run,
        extra_args=[f"--dependency=afterany:{array_id}"],
    )

    return {"array": array_id, "save": save_id, "run_dirs": run_dirs}


def validate_parallelization(nml_dict, slurm_script_path=None, slurm_ntasks=None):
    """
    Validates GENE parallelization parameters:
    1. Total number of species must be divisible by n_procs_s.
    2. n_procs_sim must equal the product of all directional n_procs_*.
    3. n_procs_sim must match the ntasks requested in the Slurm submission script.
    """
    parallel = nml_dict.get("parallelization", {})
    if not parallel:
        raise ValueError("Missing &parallelization namelist block.")

    # 1. Count total species in parameter file
    species_list = []
    if "species" in nml_dict:
        raw_species = nml_dict["species"]
        species_list.extend(
            raw_species if isinstance(raw_species, list) else [raw_species]
        )
    for key, val in nml_dict.items():
        if key.startswith("species_") and isinstance(val, dict):
            species_list.append(val)

    num_species = len(species_list)
    n_procs_s = int(parallel.get("n_procs_s", 1))

    # Check: Species count must be divisible by n_procs_s
    if num_species % n_procs_s != 0:
        raise ValueError(
            f"Parallelization error: Total species count ({num_species}) must be "
            f"divisible by n_procs_s ({n_procs_s})."
        )

    # 2. Check internal product consistency (n_procs_sim == product of all directions)
    n_procs_sim = int(parallel.get("n_procs_sim", 0))
    proc_keys = [
        "n_procs_s",
        "n_procs_v",
        "n_procs_w",
        "n_procs_x",
        "n_procs_y",
        "n_procs_z",
    ]
    calculated_sim = 1
    for pk in proc_keys:
        calculated_sim *= int(parallel.get(pk, 1))

    if n_procs_sim != calculated_sim:
        raise ValueError(
            f"Parallelization error: n_procs_sim ({n_procs_sim}) does not equal "
            f"the product of directional processes ({calculated_sim})."
        )

    # 3. Extract ntasks from Slurm script if path is provided
    if slurm_script_path:
        with open(slurm_script_path, "r") as f:
            content = f.read()
            match = re.search(r"#SBATCH\s+(-n|--ntasks=)(\d+)", content)
            if match:
                slurm_ntasks = int(match.group(2))

    # Check: n_procs_sim must match Slurm ntasks
    if slurm_ntasks is not None and n_procs_sim != int(slurm_ntasks):
        raise ValueError(
            f"Slurm mismatch: n_procs_sim ({n_procs_sim}) does not match "
            f"Slurm requested ntasks ({slurm_ntasks})."
        )

    return True
