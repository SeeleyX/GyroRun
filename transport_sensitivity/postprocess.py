# transport_sensitivity/postprocess.py
import os
import numpy as np
import pandas as pd
from parser import load_base_parameters
from pyrokinetics import Pyro
from pyrokinetics.gk_code.gene import GKInputGENE

# --- Workaround for pyrokinetics normalisation-detection fragility ----------
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

def read_fields(scan_dir):
    """
    Uses pyrokinetics to extract field amplitudes vs time for scan_dir.

    NOTE: on this pyrokinetics version, fields are separate top-level data
    variables (ds['phi'], ds['apar'], ...) rather than one 'fields'
    variable with a 'field' coordinate -- confirmed from a real run's
    ds.data_vars. Only fields actually present are returned (e.g. no
    'bpar' if that field wasn't evolved).

    Returns dict: {'time': array, <field_name>: array of field energy
    (sum |field|^2 over all non-time dims) at each timestep}
    """
    pyro = load_pyro(scan_dir)
    ds = pyro.gk_output.data

    candidate_fields = ["phi", "apar", "bpar"]
    present_fields = [f for f in candidate_fields if f in ds]

    if not present_fields:
        raise KeyError(
            f"No field variables found in pyrokinetics output for {scan_dir}. "
            f"Available data variables: {list(ds.data_vars)}"
        )

    time = ds["time"].values
    result = {"time": time}

    for name in present_fields:
        field = ds[name]
        spatial_dims = [d for d in field.dims if d != "time"]
        energy_vs_time = (np.abs(field) ** 2).sum(dim=spatial_dims)
        result[name] = energy_vs_time.values

    return result

def transport_coefficients(scan_dir, normalize_by_field=True, field_name="phi"):
    """
    Computes the particle transport coefficient for a single passive-tracer
    GENE run:

        Gamma_z = n_z * D * (grad_n_z + C_T * grad_T_z + C_u * uprim_z + C_p)

    where z indexes the passive tracer species and grad_n_z/grad_T_z/uprim_z
    are that tracer's namelist gradients (omn/omt/uprim). D is the particle
    diffusivity; C_T, C_u, C_p are dimensionless coefficients relative to D.

    Heat (Q_z) and momentum (Pi_z) channels are deliberately not computed
    yet -- their density/temperature/mass prefactors are unconfirmed
    (pending supervisor input on the exact convention). Re-add them once
    those prefactors are settled; the particle-channel pattern below
    generalizes directly (see the transport_coefficients git history /
    earlier draft for the multi-channel version).

    normalize_by_field: if True, divides flux by the field energy (from
    read_fields) at the final timestep before solving. This removes the
    common exponential growth factor present in all fluxes/fields during a
    linear run, which otherwise makes the raw D value (though not the
    C_T/C_u/C_p ratios) depend on how long the run was. Set False only
    once confirmed (via the two-run-length sanity check) that GENE's own
    nrg.dat convention already handles this.
    """
    param_file = os.path.join(scan_dir, "parameters")
    nml = load_base_parameters(param_file)

    passive_species, active_vars = parse_passive_species(nml)
    if not passive_species or not active_vars:
        raise ValueError(f"No passive species or active gradients found in {scan_dir}")

    if 'omn' not in active_vars:
        raise ValueError(
            f"Particle diffusivity D requires 'omn' to be an active gradient "
            f"(i.e. varied across the passive tracer set) in {scan_dir}, but "
            f"active_vars={active_vars}."
        )

    omega_data = read_omega(scan_dir)

    species_list = []
    if 'species' in nml:
        raw_species = nml['species']
        species_list.extend(raw_species if isinstance(raw_species, list) else [raw_species])
    for key, val in nml.items():
        if key.startswith('species_') and isinstance(val, dict):
            species_list.append(val)
    num_species = len(species_list)

    flux_data = read_nrg(scan_dir, expected_num_species=num_species)

    norm_factor = 1.0
    if normalize_by_field:
        field_data = read_fields(scan_dir)
        if field_name not in field_data:
            available = [k for k in field_data if k != "time"]
            raise KeyError(f"Field '{field_name}' not found in {scan_dir}; available: {available}")
        norm_factor = field_data[field_name][-1]

    # 'omn' (the particle-flux driver) goes first so x[0] = D
    var_order = ['omn'] + [v for v in active_vars if v != 'omn']

    A_rows = []
    b_vec = []
    for spec in passive_species:
        row = [spec[var] for var in var_order]
        row.append(1.0)
        A_rows.append(row)

        idx = spec['index']
        dens = spec['dens']
        flux = flux_data[idx]['gamma_total']
        b_vec.append(flux / (dens * norm_factor))

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
    for i, var in enumerate(var_order[1:], start=1):
        coeff_name = var_map.get(var, f"C_{var}")
        results[coeff_name] = x[i] / D if D != 0 else 0.0

    results['C_p'] = x[-1] / D if D != 0 else 0.0

    return results
