import os
import re
import copy
from pyrokinetics import Pyro, PyroScan
from parser import load_base_parameters, save_parameters
from slurm import generate_sbatch_script, submit_job, wait_for_jobs

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def apply_update(nml_dict, group, param, value):
    """Helper to update a value in the namelist dictionary."""
    if group not in nml_dict:
        nml_dict[group] = {}
    nml_dict[group][param] = value

def _scan_parameter_specs(scan_config):
    """
    Return scan_config's parameter specs as a list.

    Accepts the single-parameter form ({'param': ..., 'values': ...}) and the
    multi-parameter form ({'params': [{...}, {...}]}), so 1D scan_configs keep
    working unchanged.
    """
    specs = list(scan_config["params"]) if "params" in scan_config else [scan_config]
    if not specs:
        raise ValueError("scan_config defines no parameters to scan over")
    return specs


def create_scan(base_filepath, scan_config, output_dir_base="scans"):
    """
    PyroScan equivalent of create_1d_scan, for scans of any dimensionality.

    Every combination of the scanned values is generated (the outer product), so
    N parameters with M values each give M**N runs.

    scan_config keys:
        param/values    : a single scanned parameter (1D scan), or
        params          : list of per-parameter dicts for a multi-dimensional scan
        value_fmt       : optional format spec for values in directory names
        dir_layout      : 'flat' (default) puts every run in one directory level,
                          e.g. scan_q0_1.35_scan_shat_1.8; 'nested' gives one
                          directory level per parameter, e.g. scan_q0_1.35/scan_shat_1.8

    Per-parameter keys (used in both forms):
        param     : name of the scanned quantity; also its directory stem
        values    : list of values to scan over
        attr      : pyro attribute holding the parameter, e.g. 'local_geometry'
        location  : path to the parameter within that attribute, e.g. ['q']
                    (attr/location may be omitted if param is one of PyroScan's
                    built-in keys, e.g. 'ky' or 'electron_temp_gradient')
        func      : optional f(pyro, **func_kwargs) applied after this value is
                    set, for self-consistent updates -- the PyroScan equivalent
                    of the callable entries in create_1d_scan's values list

    Returns the list of generated run directories, as create_1d_scan does.
    """
    specs = _scan_parameter_specs(scan_config)

    layout = scan_config.get("dir_layout", "flat")
    if layout not in ("flat", "nested"):
        raise ValueError(f"dir_layout must be 'flat' or 'nested', got {layout!r}")

    # Prefixing the key gives PyroScan run directories named scan_<param>_<value>,
    # matching create_1d_scan and what postprocess.collect_scan_data expects.
    parameter_dict = {}
    for spec in specs:
        key = f"scan_{spec['param']}"
        if key in parameter_dict:
            raise ValueError(f"Parameter {spec['param']!r} is scanned more than once")
        values = list(spec["values"])
        if not values:
            raise ValueError(f"Parameter {spec['param']!r} has no values to scan over")
        parameter_dict[key] = values

    pyro = Pyro(gk_file=base_filepath)
    scan = PyroScan(
        pyro,
        parameter_dict,
        base_directory=output_dir_base,
        file_name="parameters",
        value_fmt=scan_config.get("value_fmt", ".4g"),
        parameter_separator="_" if layout == "flat" else "/",
    )

    for spec in specs:
        key = f"scan_{spec['param']}"
        if "attr" in spec:
            attr, location = spec["attr"], list(spec["location"])
        elif spec["param"] in scan.parameter_map:
            attr, location = scan.parameter_map[spec["param"]]
        else:
            raise ValueError(
                f"Unknown scan parameter {spec['param']!r}. Supply 'attr' and "
                f"'location', or use a built-in key: {sorted(scan.parameter_map)}"
            )
        scan.add_parameter_key(key, attr, location)

        if spec.get("func") is not None:
            scan.add_parameter_func(key, spec["func"], spec.get("func_kwargs", {}))

    scan.write()

    generated_dirs = [str(d) for d in scan.run_directories]
    for d in generated_dirs:
        print(f"Created: {d}")
    print(
        f"{len(generated_dirs)} runs from a {len(specs)}D scan over "
        f"{', '.join(s['param'] for s in specs)}"
    )

    return generated_dirs


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

        link_path = os.path.join(scan_dir, "gene_uprim")
        if not os.path.exists(link_path):
            os.symlink(gene_executable, link_path)
        
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
