import os
import re
from pyrokinetics import Pyro, PyroScan
from parser import load_base_parameters
from slurm import generate_sbatch_script, submit_job, wait_for_jobs

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_scan(base_filepath, scan_config, output_dir_base="scans"):
    """
    Build a scan with PyroScan. scan_config is one entry of config.yaml's
    'scans' list:

        scan:
          - parameter: q0
            attr: local_geometry
            location: [q]
            values: [1.35, 1.5, 1.65]
          - parameter: ky
            values: [0.1, 0.3]

    Every combination of the values is generated. attr and location register
    where the quantity lives in the pyro object, and are only needed for
    parameters PyroScan does not already define.
    """
    parameters = scan_config["scan"]

    pyro = Pyro(gk_file=base_filepath)
    scan = PyroScan(
        pyro,
        {p["parameter"]: p["values"] for p in parameters},
        base_directory=output_dir_base,
    )

    for p in parameters:
        if "attr" in p:
            scan.add_parameter_key(p["parameter"], p["attr"], p["location"])

    scan.write()

    return [str(d) for d in scan.run_directories]


def execute_scan(
    base_filepath,
    scan_config,
    output_dir_base="scans",
    template_path=None,
    gene_executable=None,
    poll_interval=30,
    dry_run=True,
    slurm_kwargs=None,
    wait=False,
):

    if template_path is None:
        template_path = os.path.join(PROJECT_ROOT, "templates", "slurm_template.sh")
    if gene_executable is None:
        raise ValueError("gene_executable path must be provided")

    scan_dirs = create_scan(base_filepath, scan_config, output_dir_base=output_dir_base)
    active_jobs = {}

    for scan_dir in scan_dirs:
        folder_name = os.path.basename(scan_dir)
        param_file = os.path.join(scan_dir, "input.gene")
        nml = load_base_parameters(param_file)

        link_path = os.path.join(scan_dir, "gene_uprim")
        if not os.path.exists(link_path):
            os.symlink(gene_executable, link_path)

        # 1. Extract n_procs_sim dynamically from generated parameters
        n_procs_sim = int(nml.get("parallelization", {}).get("n_procs_sim", 64))

        # 2. Run parallelization checks BEFORE generating script / submitting
        validate_parallelization(nml, slurm_ntasks=n_procs_sim)

        # 3. Populate template values
        output_script = os.path.join(scan_dir, "run_gene.sh")
        context = {
            "job_name": folder_name,
            "ntasks": n_procs_sim,
            "time_limit": "02:00:00",
            "partition": "standard",
            "account": "default_account",
            "run_dir": os.path.abspath(scan_dir),
            "gene_executable": gene_executable,
        }
        if slurm_kwargs:
            context.update(slurm_kwargs)

        # 4. Generate submit script from template and submit job
        script_path = generate_sbatch_script(template_path, output_script, context)
        job_id = submit_job(script_path, scan_dir, dry_run=dry_run)
        active_jobs[job_id] = scan_dir

    print(f"\nSuccessfully submitted {len(active_jobs)} jobs.")

    if wait:
        # Pause execution until all Slurm tasks finish
        wait_for_jobs(active_jobs, poll_interval=poll_interval, dry_run=dry_run)

    return active_jobs


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
