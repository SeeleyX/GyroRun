# transport_sensitivity/postprocess.py
import os
import numpy as np
import pandas as pd
from parser import load_base_parameters
from pyrokinetics import Pyro
from pyrokinetics.gk_code.gene import GKInputGENE

# --- Workaround for pyrokinetics normalisation-detection fragility ----------
# Two separate issues observed on real GENE files from this pipeline:
#   1. Auto-detection raises ValueError on artificial/reduced electron mass
#      ratios (common GENE speed-up trick, e.g. me/mi = 1/400).
#   2. Even when detection succeeds but labels the run a "bespoke" convention
#      (any reference value differing from pyrokinetics' hardcoded defaults),
#      pyro.load_gk_output() re-parses the input file internally via a
#      separate code path that never registers that bespoke convention on
#      the norms object it then looks it up on -- an internal pyrokinetics
#      inconsistency, not a problem with the input file.
#
# Rather than patch each new edge case as it's discovered, we skip
# convention *detection* entirely and always use pyrokinetics' plain
# default convention. This does NOT change any of the raw values read from
# nrg/omega/field files -- it only affects how pyrokinetics *labels* and
# would convert those values into other unit systems. Since our analysis
# only ever compares values within the same run's own normalization
# (fluxes and gradients from the same 'parameters' file), this is safe for
# our purposes. It matters only if you later want pyrokinetics to convert
# output into absolute SI-like units against a *different* code's file --
# this pipeline isn't doing that.
def _skip_normalisation_detection(self):
    self.norm_convention = "pyrokinetics"
    self._convention_dict = {}

GKInputGENE._detect_normalisation = _skip_normalisation_detection
# -----------------------------------------------------------------------------


def parse_passive_species(nml_dict):
    """
    Extracts passive species metadata and tracks their 1-based index for
    flux column/species mapping.

    NOTE: kept on f90nml (not pyrokinetics) deliberately. The 'passive'
    flag and the raw omn/omt/uprim gradient values are GENE-specific
    normalized-namelist quantities; pyrokinetics standardizes species
    data into its own reference-unit system and does not expose GENE's
    'passive' flag, so mixing the two here would silently corrupt units.
    """
    all_species = []

    if 'species' in nml_dict:
        raw_species = nml_dict['species']
        if isinstance(raw_species, list):
            all_species.extend(raw_species)
        elif isinstance(raw_species, dict):
            all_species.append(raw_species)

    for key, val in nml_dict.items():
        if key.startswith('species_') and isinstance(val, dict):
            all_species.append(val)

    passive_species = []

    for idx, spec in enumerate(all_species, start=1):
        if spec.get('passive', False):
            species_data = {
                'index': idx,
                'name': spec.get('name', f"species_{idx}"),
                'dens': float(spec.get('dens', 1.0)),
                'omn': float(spec.get('omn', 0.0)),
                'omt': float(spec.get('omt', 0.0)),
                'uprim': float(spec.get('uprim', 0.0)),
            }
            passive_species.append(species_data)

    active_vars = []
    for var in ['omn', 'omt', 'uprim']:
        if any(abs(s[var]) > 1e-12 for s in passive_species):
            active_vars.append(var)

    return passive_species, active_vars


def load_pyro(scan_dir):
    """
    Loads a Pyro object for the GENE run in scan_dir and reads its
    gyrokinetic output (omega/eigenvalues + fluxes). Pyrokinetics locates
    the correct associated output files (nrg, omega/eigenvalues, field)
    itself, rather than us guessing filenames.
    """
    param_file = os.path.join(scan_dir, "parameters")
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"No 'parameters' file in {scan_dir}")

    pyro = Pyro(gk_file=param_file, gk_code="GENE")
    pyro.load_gk_output()
    return pyro


def read_omega(scan_dir):
    """
    Uses pyrokinetics to extract growth rate and mode frequency at the
    final timestep, for the single-ky flux-tube run in scan_dir.
 
    Returns dict with 'ky', 'gamma' (growth rate), 'omega' (real frequency).
    """
    pyro = load_pyro(scan_dir)
    ds = pyro.gk_output.data
 
    eig = ds["eigenvalues"]
    # Reduce to the final timestep along the time dimension
    eig_final = eig.isel(time=-1)
 
    # For a single-ky linear flux-tube run there should be one ky value;
    # take it regardless of whether ky/kx are length-1 dims or scalars.
    eig_final = eig_final.squeeze()
 
    ky_val = float(ds["ky"].values.squeeze()) if "ky" in ds else float(eig_final["ky"].values)
    # eigenvalue convention is (real, imag) = (mode frequency, growth rate) --
    # confirmed against raw omega.dat: real part matches omega, imag matches gamma.
    omega = float(np.real(eig_final.values))
    gamma = float(np.imag(eig_final.values))
 
    return {"ky": ky_val, "gamma": gamma, "omega": omega}


def read_nrg(scan_dir, expected_num_species=None):
    """
    Uses pyrokinetics to extract particle, heat, and momentum fluxes per
    species at the final timestep, each split into electrostatic (es) and
    electromagnetic (em) contributions, plus their sum (total).
 
    Returns dict mapping species 1-based index -> {
        'gamma_es', 'gamma_em', 'gamma_total',      # particle flux
        'q_es', 'q_em', 'q_total',                  # heat flux
        'pi_es', 'pi_em', 'pi_total',                # momentum flux
    }
    matching the ORDER species appear in pyro.gk_output.data['species']
    (assumed same order as the 'parameters' file -- see WARNING check below).
 
    expected_num_species: optional. If provided (e.g. from the namelist's
    own species count via parse_passive_species/nml), a mismatch against
    pyrokinetics' own species count is flagged -- catches cases where the
    'parameters' file and pyrokinetics disagree on how many species exist,
    which would otherwise silently misalign the species-index mapping used
    by analyze_single_run.
 
    VERIFY: field/dim names may differ by pyrokinetics version -- check
    pyro.gk_output.data against a known-good run before trusting this on
    a full scan.
    """
    pyro = load_pyro(scan_dir)
    ds = pyro.gk_output.data
 
    # (data variable name, short key prefix used in the returned dict)
    flux_vars = [
        ("particle", "gamma"),
        ("heat", "q"),
        ("momentum", "pi"),
    ]
 
    missing = [name for name, _ in flux_vars if name not in ds]
    if missing:
        raise KeyError(
            f"Flux variable(s) {missing} not found in pyrokinetics output for "
            f"{scan_dir}. Available data variables: {list(ds.data_vars)}"
        )
 
    species_names = [str(s) for s in ds["species"].values]
 
    fluxes = {idx: {} for idx in range(1, len(species_names) + 1)}
 
    for var_name, key_prefix in flux_vars:
        flux = ds[var_name].isel(time=-1)
 
        # Sum over field (electrostatic 'phi' + electromagnetic 'apar'/'bpar')
        # if a 'field' dimension exists, so we can report es/em/total.
        if "field" in flux.dims:
            field_names = [str(f) for f in flux["field"].values]
            es_fields = [f for f in field_names if f.lower() in ("phi", "es", "electrostatic")]
            em_fields = [f for f in field_names if f not in es_fields]
 
            es_by_species = flux.sel(field=es_fields).sum("field") if es_fields else flux.isel(field=0) * 0
            em_by_species = flux.sel(field=em_fields).sum("field") if em_fields else flux.isel(field=0) * 0
        else:
            es_by_species = flux
            em_by_species = flux * 0
 
        for idx, name in enumerate(species_names, start=1):
            es_val = float(es_by_species.sel(species=name).values.squeeze())
            em_val = float(em_by_species.sel(species=name).values.squeeze())
            fluxes[idx][f"{key_prefix}_es"] = es_val
            fluxes[idx][f"{key_prefix}_em"] = em_val
            fluxes[idx][f"{key_prefix}_total"] = es_val + em_val
 
    if expected_num_species is not None and len(fluxes) != expected_num_species:
        print(
            f"WARNING in {scan_dir}: pyrokinetics reported {len(fluxes)} species "
            f"({species_names}) but namelist parsing expected {expected_num_species}. "
            f"Species-index mapping between parse_passive_species and read_nrg may "
            f"be misaligned -- verify species ORDER matches between the 'parameters' "
            f"file and pyro.gk_output.data['species'] before trusting these numbers."
        )
 
    return fluxes


def analyze_single_run(scan_dir, flux_type="gamma_total"):
    """
    Analyzes a single GENE simulation folder to compute transport coefficients.
    Acts as the `analysis_func` passed to `collect_scan_data`.

    UNIT-CONSISTENCY WARNING: 'omn'/'omt'/'uprim'/'dens' below are read raw
    from the namelist (GENE-normalized units), while flux_data now comes
    from pyrokinetics (pyro-reference-normalized units). If these two unit
    systems don't coincide for your setup, D and the C_* coefficients will
    be silently wrong. Confirm this on one hand-checked run -- e.g. compare
    D from this pipeline against the OLD f90nml-based nrg.dat parsing for
    the same scan_dir -- before trusting a full scan's results.
    """
    param_file = os.path.join(scan_dir, "parameters")
    nml = load_base_parameters(param_file)

    passive_species, active_vars = parse_passive_species(nml)
    if not passive_species or not active_vars:
        raise ValueError(f"No passive species or active gradients found in {scan_dir}")

    omega_data = read_omega(scan_dir)
    num_species = len(nml.get('species', [])) if 'species' in nml else 0
    if num_species == 0:
        num_species = sum(1 for k in nml if k.startswith('species'))

    flux_data = read_nrg(scan_dir, num_species=num_species)

    A_rows = []
    b_vec = []

    for spec in passive_species:
        idx = spec['index']
        dens = spec['dens']

        row = [spec[var] for var in active_vars]
        row.append(1.0)

        A_rows.append(row)

        flux = flux_data[idx][flux_type]
        b_vec.append(flux / dens)

    A = np.array(A_rows)
    b = np.array(b_vec)

    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    D = x[0]
    results = {
        'ky': omega_data['ky'],
        'gamma': omega_data['gamma'],
        'omega': omega_data['omega'],
        'D': D,
    }

    var_map = {'omn': 'C_n', 'omt': 'C_T', 'uprim': 'C_u'}
    for i, var in enumerate(active_vars[1:], start=1):
        coeff_name = var_map.get(var, f"C_{var}")
        results[coeff_name] = x[i] / D if D != 0 else 0.0

    results['C_p'] = x[-1] / D if D != 0 else 0.0

    return results


def collect_scan_data(scan_dirs, analysis_func, base_filepath=None):
    """
    Iterates through completed scan directories, automatically extracts any
    input parameters that changed relative to base_filepath, and appends
    physics results from analysis_func.
    """
    base_nml = load_base_parameters(base_filepath) if base_filepath and os.path.exists(base_filepath) else {}
    results = []

    for scan_dir in scan_dirs:
        param_file = os.path.join(scan_dir, "parameters")
        if not os.path.exists(param_file):
            print(f"Skipping {scan_dir}: No parameters file found.")
            continue

        nml = load_base_parameters(param_file)
        row_data = {'scan_dir': scan_dir, 'folder': os.path.basename(scan_dir)}

        if base_nml:
            for group, params in nml.items():
                if isinstance(params, dict):
                    base_group = base_nml.get(group, {})
                    for key, val in params.items():
                        if isinstance(base_group, dict) and base_group.get(key) != val:
                            row_data[key] = val

        try:
            physics_outputs = analysis_func(scan_dir)
            row_data.update(physics_outputs)
        except Exception as e:
            print(f"Error analyzing {scan_dir}: {e}")

        results.append(row_data)

    return pd.DataFrame(results)