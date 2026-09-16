import os
import re
import copy
from parser import load_base_parameters, save_parameters
from slurm import generate_sbatch_script, submit_job, wait_for_jobs

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def apply_update(nml_dict, group, param, value):
    """Helper to update a value in the namelist dictionary."""
    if group not in nml_dict:
        nml_dict[group] = {}
    nml_dict[group][param] = value

def create_1d_scan(base_filepath, scan_config, output_dir_base="scans"):
    base_nml = load_base_parameters(base_filepath)
    os.makedirs(output_dir_base, exist_ok=True)
    
    generated_dirs = []
    
    for i, val in enumerate(scan_config['values']):
        new_nml = copy.deepcopy(base_nml)
        
        # Check if the value is a custom function
        if callable(val):
            updates = val(base_nml) 
            for group, param, calculated_val in updates:
                apply_update(new_nml, group, param, calculated_val)
            # Use provided label if it exists, otherwise use an index
            label = scan_config.get('labels', [])[i] if 'labels' in scan_config else f"run_{i}"
            dir_name = f"scan_{scan_config['param']}_{label}"
        else:
            # Standard single-variable update
            apply_update(new_nml, scan_config['group'], scan_config['param'], val)
            dir_name = f"scan_{scan_config['param']}_{val}"
            
        full_dir_path = os.path.join(output_dir_base, dir_name)
        os.makedirs(full_dir_path, exist_ok=True)
        
        save_parameters(new_nml, os.path.join(full_dir_path, "parameters"))
        generated_dirs.append(full_dir_path)
        print(f"Created: {full_dir_path}")
        
    return generated_dirs

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

    scan_dirs = create_1d_scan(base_filepath, scan_config, output_dir_base=output_dir_base)
    active_jobs = {}
    
    for scan_dir in scan_dirs:
        folder_name = os.path.basename(scan_dir)
        param_file = os.path.join(scan_dir, "parameters")
        nml = load_base_parameters(param_file)
        
        # 1. Extract n_procs_sim dynamically from generated parameters
        n_procs_sim = int(nml.get('parallelization', {}).get('n_procs_sim', 64))
        
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
            "gene_executable": gene_executable
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
    parallel = nml_dict.get('parallelization', {})
    if not parallel:
        raise ValueError("Missing &parallelization namelist block.")

    # 1. Count total species in parameter file
    species_list = []
    if 'species' in nml_dict:
        raw_species = nml_dict['species']
        species_list.extend(raw_species if isinstance(raw_species, list) else [raw_species])
    for key, val in nml_dict.items():
        if key.startswith('species_') and isinstance(val, dict):
            species_list.append(val)
            
    num_species = len(species_list)
    n_procs_s = int(parallel.get('n_procs_s', 1))

    # Check: Species count must be divisible by n_procs_s
    if num_species % n_procs_s != 0:
        raise ValueError(
            f"Parallelization error: Total species count ({num_species}) must be "
            f"divisible by n_procs_s ({n_procs_s})."
        )

    # 2. Check internal product consistency (n_procs_sim == product of all directions)
    n_procs_sim = int(parallel.get('n_procs_sim', 0))
    proc_keys = ['n_procs_s', 'n_procs_v', 'n_procs_w', 'n_procs_x', 'n_procs_y', 'n_procs_z']
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
        with open(slurm_script_path, 'r') as f:
            content = f.read()
            match = re.search(r'#SBATCH\s+(-n|--ntasks=)(\d+)', content)
            if match:
                slurm_ntasks = int(match.group(2))

    # Check: n_procs_sim must match Slurm ntasks
    if slurm_ntasks is not None and n_procs_sim != int(slurm_ntasks):
        raise ValueError(
            f"Slurm mismatch: n_procs_sim ({n_procs_sim}) does not match "
            f"Slurm requested ntasks ({slurm_ntasks})."
        )

    return True