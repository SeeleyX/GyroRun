# Architecture Overview

GyroRun is a Python command-line package for creating multidimensional GENE
gyrokinetic parameter scans, submitting each generated run to Slurm, and
post-processing completed runs. This document describes the repository as it
currently operates and should be updated alongside architectural changes.

## 1. Project Structure

```text
GyroRun/
├── config.yaml                 # Paths, Slurm settings, run name, and scans
├── templates/                  # Base GENE input and Slurm script templates
├── src/gyrorun/
│   ├── run_scan.py             # YAML entry point and scan orchestration
│   ├── builder.py              # PyroScan creation and per-run submission setup
│   ├── slurm.py                # Slurm script rendering, submission, and polling
│   ├── parser.py               # Fortran namelist read/write helpers
│   └── postprocess.py          # GENE output parsing and scan result collection
├── tests/                      # Pytest coverage for builder, Slurm, and analysis
├── pyproject.toml              # Python package metadata and dependencies
└── uv.lock                     # Locked Python dependency set
```

## 2. System Flow

```text
config.yaml
    │
    ▼
gyrorun.run_scan.main
    │ resolves repository-relative paths and iterates scans
    ▼
gyrorun.builder.create_scan
    │ creates a Pyro object and a Cartesian-product PyroScan
    ▼
generated GENE input directories
    │
    ▼
gyrorun.builder.execute_scan ──► rendered Slurm scripts ──► sbatch/GENE
                                                              │
                                                              ▼
                                                gyrorun.postprocess results
```

`run_scan.main` loads the root configuration, resolves paths relative to the
repository, appends `run.name` to `paths.output_dir_base`, and sends each item
in `scans` to `execute_scan`. `create_scan` builds a `pyrokinetics.PyroScan`
from a base GENE parameter file; each parameter's values form a Cartesian
product and each combination receives a run directory and input file.

## 3. Core Components

### Configuration and scan generation

Each `scans` item has a `scan` list containing parameter mappings and an
optional scan-local `flags` mapping:

```yaml
- scan:
    - parameter: q0
      attr: local_geometry
      location: [q]
      values: [1.35, 1.5]
    - flags:
        - enforce_consistent_beta_prime
```

`parameter` and `values` define the scan dimension. `attr` and `location`
locate values that PyroScan does not provide as built-in parameter keys.
`enforce_consistent_beta_prime` is currently a reserved no-op hook in
`create_scan`; it is intentionally scoped to the scan that declares it and
will later contain the required Pyrokinetics call.

### Slurm execution

For every generated directory, `execute_scan` reads its namelist, verifies
GENE parallelisation settings, creates a `gene_uprim` symlink, renders
`templates/slurm_template.sh`, and submits it through `sbatch`. `dry_run`
returns synthetic job IDs instead of submitting jobs. The Slurm module can
also poll submitted job IDs with `sacct`.

### Post-processing

`postprocess.py` reads generated GENE `parameters`, `omega`, and `nrg` output
through Pyrokinetics and f90nml. It identifies passive species, derives
transport coefficients by least-squares fitting, and combines per-directory
analysis results in a pandas DataFrame.

## 4. Dependencies and External Systems

- Python 3.14+ package managed with uv.
- Pyrokinetics represents GENE inputs and creates N-dimensional scans.
- GENE is supplied by the absolute `paths.gene_executable` configuration path.
- Slurm (`sbatch` and `sacct`) runs and monitors simulations; the cluster
  partition, account, and time limit are configured in `config.yaml`.
- GENE input and Slurm templates are repository-local files under `templates/`.

No web service, database, authentication layer, or CI configuration is
present in this repository.

## 5. Development and Testing

Tests use pytest and cover parallelisation validation, Slurm rendering and
polling, post-processing, and the dry-run builder pipeline. Run them with:

```bash
uv run pytest -q
```

The configured executable and Slurm settings are environment-specific. Use
`run.dry_run: true` when exercising scan generation without a cluster
submission.
