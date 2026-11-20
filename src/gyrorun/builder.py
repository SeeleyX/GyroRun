import json
import os
import re
import sys
import numpy as np
from pyrokinetics import Pyro, PyroScan, PyroHypercube
from pyrokinetics.pyroscan import get_from_dict, set_in_dict
from scipy.stats import qmc
from .slurm import generate_sbatch_script, submit_job

# repo root, two levels above src/gyrorun/
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def create_scan(base_filepath, scan_config, output_dir_base="scans", gk_code=None):
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
          - units: tglf

    A scan generates every combination of the values. A cube draws n_samples
    points from a Latin hypercube over the [min, max] range in each
    parameter's values ('scale: log' on a parameter draws it uniformly in
    log(value) instead). attr and location register where the quantity lives in
    the pyro object, and are only needed for parameters pyrokinetics does not
    already define. Flags apply only to the scan that declares them.

    units names the pyro normalisation convention the values are written in
    (pyrokinetics, gs2, cgyro, gene, tglf, gftm, ...), whatever code is being
    run: pyro converts them to the code it writes. Without it the values are
    plain floats and PyroScan attaches the base input's own units, with a
    warning. 'value_fmt: ".3f"' sets the precision of the values in run directory
    names. pyro widens it on reload if the values, in the reloading unit, collide
    when rounded, which then no longer finds the directories that were written:
    pick one wide enough that neither happens. A parameter item may carry its own 'units:', overriding the scan's;
    'units: null' leaves that parameter's values plain, i.e. in the base input's
    own convention.

    gk_code (or the scan's own 'gk_code' item, which wins) converts the scan
    to that code before writing: scan.convert_gk_code, so the decks are the
    new code's default file name. 'code_flags: {KEY: value}' items are set on
    every run's Pyro (pyro.add_flags) after conversion: code settings, not scan
    parameters. A cube may instead read an existing per-case database:

        cube:
          - from_directory:
              root: $GYRO_DATA_OUTPUT/GS2/Runs/kbm/M1   # env vars expanded
              pattern: "*"
              params: [ky, beta]
              gk_code: GS2                              # code of the source decks
          - gk_code: GFTM
          - code_flags: {WIDTH: 0.6, FILTER: 0.5}
          - code_flags_per_sample: widths.json      # {"case_0": {"WIDTH": 0.7}, ...}

    one sample per matching run directory, keeping that run's own values.
    code_flags_per_sample names a JSON file mapping each run name to its own
    flags, applied after code_flags (so they win). It must list exactly the
    scan's runs: a missing or extra name is an error, never a silent default.

    A cube may also take its samples from a table instead of drawing them, to
    continue an existing design: one line per parameter, 'name v1 v2 ...'.
    A parameter item's 'column' names its row when that differs from its name:

        cube:
          - samples_from: {file: $GYRO_DATA_OUTPUT/.../params.in, start: 300, stop: 1000}
          - parameter: ion_temp_gradient
            column: deuterium_temp_gradient
            attr: local_species
            location: [ion1, inverse_lt]
            link: {species: [ion2], mode: equal}

    'link' copies a species parameter to other species: mode 'equal' sets
    them to the same value (gradients); 'scale' keeps each one's ratio to it
    from the base input (collision frequencies). Links, then the flags
    'enforce_quasineutrality: <species>' (pyro's LocalSpecies method),
    enforce_consistent_beta_prime and enforce_consistent_pvg, run once per
    run after every parameter is set, so they see that run's final values.
    """
    kind = "cube" if "cube" in scan_config else "scan"
    items = scan_config[kind]
    values = {item["parameter"]: item.get("values") for item in items if "parameter" in item}
    flags = [flag for item in items for flag in item.get("flags", [])]

    source = next((i["from_directory"] for i in items if "from_directory" in i), None)
    gk_code = next((i["gk_code"] for i in items if "gk_code" in i), gk_code)
    code_flags = {k: v for i in items for k, v in i.get("code_flags", {}).items()}

    if source:
        root = os.path.expandvars(os.path.expanduser(source["root"]))
        # from_directory sets base_directory to the source; write() redirects it
        scan = PyroHypercube.from_directory(**{**source, "root": root})
    else:
        pyro = Pyro(gk_file=base_filepath)
        table = next((i["samples_from"] for i in items if "samples_from" in i), None)
        if table:
            with open(os.path.expandvars(os.path.expanduser(table["file"]))) as f:
                rows = {line.split()[0]: line.split()[1:] for line in f if line.strip()}
            column = {i["parameter"]: i.get("column", i["parameter"]) for i in items if "parameter" in i}
            cut = slice(table.get("start", 0), table.get("stop"))
            values = {name: [float(v) for v in rows[column[name]][cut]] for name in values}
        elif kind == "cube":
            n_samples = next(item["n_samples"] for item in items if "n_samples" in item)
            seed = next((f["seed"] for f in flags if isinstance(f, dict) and "seed" in f), None)
            # 'scale: log' on a parameter draws it uniformly in log(value): the same number of samples per decade
            scale = [item.get("scale", "linear") for item in items if "parameter" in item]
            if set(scale) - {"linear", "log"}:
                raise ValueError(f"scale {sorted(set(scale) - {'linear', 'log'})}: use 'linear' or 'log'")
            log = np.array(scale) == "log"
            bounds = np.array(list(values.values()), float)
            if (bounds[log] <= 0).any():
                raise ValueError("scale: log needs a range above zero")
            bounds[log] = np.log(bounds[log])
            # pyro has no sampler of its own; its how-to draws samples by hand too
            sample = qmc.LatinHypercube(d=len(values), rng=seed).random(n_samples)
            sample = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
            sample[:, log] = np.exp(sample[:, log])
            values = {name: sample[:, i].tolist() for i, name in enumerate(values)}
        convention = next((i["units"] for i in items if "units" in i and "parameter" not in i), None)
        if convention is not None or any("units" in i for i in items):
            values = attach_units(pyro, values, items, convention)
        cls = PyroHypercube if kind == "cube" else PyroScan
        fmt = {"value_fmt": i["value_fmt"] for i in items if "value_fmt" in i}
        scan = cls(pyro, values, base_directory=output_dir_base, **fmt)

    for item in items:
        if "parameter" in item and "attr" in item:
            scan.add_parameter_key(item["parameter"], item["attr"], item["location"])

    # pyro runs a parameter's func right after setting that parameter, so funcs that
    # depend on several parameters all go on the last one: they see the run's final values
    post = []
    for item in items:
        if "link" in item:
            attr, (primary, *key) = scan.parameter_map[item["parameter"]]
            mode = item["link"].get("mode", "equal")
            if mode not in ("equal", "scale"):
                raise ValueError(f"link mode {mode!r}: use 'equal' or 'scale'")
            base = getattr(scan.base_pyro, attr)
            for other in item["link"]["species"]:
                ratio = get_from_dict(base, [other, *key]) / get_from_dict(base, [primary, *key])
                factor = 1 if mode == "equal" else ratio
                post.append(lambda p, a=attr, o=other, f=factor, s=primary, k=key: set_in_dict(
                    getattr(p, a), [o, *k], f * get_from_dict(getattr(p, a), [s, *k])))
    qn = next((f["enforce_quasineutrality"] for f in flags if isinstance(f, dict) and "enforce_quasineutrality" in f), None)
    if qn:
        post.append(lambda p: p.local_species.enforce_quasineutrality(qn))
    if "enforce_consistent_beta_prime" in flags and "beta" in values:
        post.append(lambda p: p.enforce_consistent_beta_prime())
    if "enforce_consistent_pvg" in flags and "gamma_exb" in values:
        post.append(Pyro.enforce_consistent_pvg)
    if post:
        scan.add_parameter_func(list(scan.parameter_dict)[-1], lambda p: [f(p) for f in post], {})

    if gk_code:
        scan.convert_gk_code(gk_code)
    if code_flags:
        for pyro in scan.pyro_dict.values():
            pyro.add_flags(code_flags)
    per_sample = next((i["code_flags_per_sample"] for i in items if "code_flags_per_sample" in i), None)
    if per_sample:
        with open(os.path.expandvars(os.path.expanduser(per_sample))) as f:
            table = json.load(f)
        if set(table) != set(scan.pyro_dict):
            raise ValueError(
                f"code_flags_per_sample: runs without flags {sorted(set(scan.pyro_dict) - set(table))[:5]}, "
                f"flags without a run {sorted(set(table) - set(scan.pyro_dict))[:5]}"
            )
        for name, pyro in scan.pyro_dict.items():
            pyro.add_flags(table[name])
        # Record each flag as an ordinary per-sample parameter, so pyroscan.json
        # keeps what every run used and PyroScan.sample_pyro re-applies it.
        # The Pyro attribute path is gk_input.data[<flag>]; flags are flat for
        # GFTM/TGLF, which is all this supports.
        keys = sorted({k for flags in table.values() for k in flags})
        if any(set(flags) != set(keys) for flags in table.values()):
            raise ValueError("code_flags_per_sample: every run must carry the same flags")
        scan.set_parameter_dict(
            {**scan.parameter_dict, **{k: [table[n][k] for n in scan.pyro_dict] for k in keys}}
        )
        # add_parameter_key reads the base value (for units), so the key must exist
        # there; the base then carries the first run's value, which every sample overrides
        scan.base_pyro.add_flags({k: table[next(iter(scan.pyro_dict))][k] for k in keys})
        stored = {k.lower(): k for k in scan.base_pyro.gk_input.data}  # GFTM stores keys lower case
        for k in keys:
            scan.add_parameter_key(k, "gk_input", ["data", stored[k.lower()]])
    scan.write(base_directory=output_dir_base)

    return [str(d) for d in scan.run_directories]


def attach_units(pyro, values, items, convention):
    """
    Attach to each parameter's values its unit in the named convention, taken
    from the base pyro: the unit of its current value, converted to
    pyro.norms.<convention>. Dimensionless parameters stay plain.
    """
    parameter_map = PyroScan(pyro, {}).parameter_map
    parameter_map.update(
        {item["parameter"]: [item["attr"], item["location"]] for item in items if "attr" in item}
    )
    own = {i["parameter"]: i["units"] for i in items if "parameter" in i and "units" in i}
    out = {}
    for name, vals in values.items():
        units = own.get(name, convention)
        if units is None:
            out[name] = vals
            continue
        try:
            norm = getattr(pyro.norms, units)
        except AttributeError:
            raise ValueError(f"units: {units!r} is not a pyro normalisation convention")
        attr, location = parameter_map[name]
        current = get_from_dict(getattr(pyro, attr), location[:-1])[location[-1]]
        if not hasattr(current, "units") or current.dimensionless:
            out[name] = vals
            continue
        try:
            unit = current.to(norm).units
        except Exception as e:
            raise ValueError(
                f"pyro cannot express {name} ({current.units}) in the {units} "
                f"convention for this base input: {e}"
            ) from e
        out[name] = np.asarray(vals) * unit
    return out


def execute_scan(
    base_filepath,
    scan_config,
    slurm_config,
    output_dir_base="scans",
    array_template=None,
    save_template=None,
    dry_run=True,
    gk_code=None,
    tearing_parameter=False,
    store_dir=None,
):
    """
    Write the scan, then submit one Slurm array job over its run directories
    and a save job that runs after it to write pyroscan.nc. Does not wait.

    slurm_config is config.yaml's 'slurm' mapping. run_command is required;
    nodes, ntasks, max_parallel, qos, setup and save_time_limit are optional.
    tearing_parameter: the save job stores pyro's tearing parameter and A_par
    parity (linear runs only).
    store_dir: the save job writes pyroscan.nc and its sidecars here, not in the
    run leaf (None = in the leaf).
    """
    if array_template is None:
        array_template = os.path.join(PROJECT_ROOT, "templates", "slurm_array.sh")
    if save_template is None:
        save_template = os.path.join(PROJECT_ROOT, "templates", "slurm_save.sh")

    run_dirs = create_scan(
        base_filepath, scan_config, output_dir_base=output_dir_base, gk_code=gk_code
    )
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
        "qos_line": f"#SBATCH --qos={slurm_config['qos']}" if slurm_config.get("qos") else "",
        "setup": "\n".join(slurm_config.get("setup", [])),
        "last_index": len(run_dirs) - 1,
        "runs_file": runs_file,
        "scan_dir": scan_dir,
        "save_flags": " ".join(
            (["--tearing"] if tearing_parameter else []) + (["--store", store_dir] if store_dir else [])
        ),
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
