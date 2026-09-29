# Architecture Overview

GyroRun is a Python command-line package for creating multidimensional
gyrokinetic parameter scans and Latin hypercube samples, submitting them to
Slurm as one array job per scan, saving each scan's output to netCDF, and
post-processing completed runs. This document describes the repository as it
currently operates and should be updated alongside architectural changes.

## 1. Project Structure

```text
GyroRun/
├── config.yaml                 # Paths, Slurm settings, run name, and scans
├── examples/pyrocube.yaml      # Minimal runnable PyroCube example
├── templates/                  # Slurm array and save job templates
├── src/gyrorun/
│   ├── run_scan.py             # `gyrorun config.yaml` entry point
│   ├── builder.py              # PyroScan/PyroHypercube creation and submission
│   ├── layout.py               # Run-directory layout and manifest.json
│   ├── save.py                 # Reload a scan and write pyroscan.nc
│   ├── slurm.py                # Slurm script rendering and submission
│   ├── parser.py               # Fortran namelist read/write helpers
│   └── postprocess.py          # GENE output parsing and scan result collection
├── tests/                      # Pytest coverage; tests/data holds pyro's public
│                               # CGYRO_linear_scan output
├── pyproject.toml              # Package metadata; pins pyrokinetics by git branch
└── uv.lock                     # Locked dependency set (pins the pyro commit)
```

## 2. System Flow

```text
config.yaml
    │  gyrorun.run_scan.main: paths relative to the config file, $VARS expanded
    ▼
gyrorun.builder.create_scan ──► PyroScan (scan:) or PyroHypercube (cube:)
    │                            .write(): pyroscan.json + one input per run
    ▼
gyrorun.builder.execute_scan
    ├─► runs.txt (one run directory per line)
    ├─► slurm_array.sh ── sbatch ──► array job, task i runs in line i+1
    └─► slurm_save.sh  ── sbatch --dependency=afterany:<array> ──►
                            python -m gyrorun.save <scan_dir> ──► pyroscan.nc
```

GyroRun is code-agnostic: pyrokinetics writes every input and reads every
output, and the code is inferred from `paths.base_input`. The only
code-specific setting is `slurm.run_command`. Each entry in `scans` is
written to `<root>/<CODE>/Runs/<run.scan_name>/<run.case>/<name>` (the
Gyrokinetic_Simulations layout; `root` is `$GYRO_DATA_OUTPUT`, or
`paths.output_dir_base` for dry runs and tests) and submitted as one array
job plus one save job. `layout.compose_leaf` builds the path from the code pyro
reads in the base input, checks each segment against `[A-Za-z0-9._-]`, and
refuses an existing non-empty leaf. `layout.write_manifest` writes
`manifest.json` in each leaf: GyroRun commit, resolved config, executable
size/mtime (if `paths.executable`), submission time, array and save job IDs.
The leaves are printed at the end for the run catalogue. Nothing polls Slurm; the save job waits on the array
job through its dependency.

## 3. Core Components

### Configuration and scan generation

Each `scans` item is a `scan` (outer product) or a `cube` (Latin hypercube)
list of parameter mappings plus an optional scan-local `flags` mapping:

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
locate values that pyrokinetics does not provide as built-in parameter keys.
`enforce_consistent_beta_prime` (with a `beta` parameter) and
`enforce_consistent_pvg` (with `gamma_exb`) register the matching pyro
consistency function for that scan only.

### Generating a PyroCube

A `cube` entry takes `n_samples`, a `[min, max]` range as each parameter's
`values`, and an optional `seed` flag:

```yaml
- cube:
    - n_samples: 300
    - parameter: ky
      values: [0.1, 1.0]
    - parameter: shat
      attr: local_geometry
      location: [shat]
      values: [0.5, 3.0]
    - flags:
        - seed: 0
```

Pyrokinetics has no sampler, so GyroRun draws the samples with
`scipy.stats.qmc.LatinHypercube(d, rng=seed)` and scales them to the ranges.
The draw is passed to `PyroHypercube`, which writes run directories
`sample_0000`, `sample_0001`, …. The same seed gives the same samples. See
`examples/pyrocube.yaml` for a runnable dry-run example:

```bash
uv run gyrorun examples/pyrocube.yaml
```

### Slurm execution

`slurm` in the config sets `partition`, `account`, `time_limit` and the
required `run_command`, plus optional `nodes` (1), `ntasks` (1),
`max_parallel` (50, the `%` limit on the array), `setup` (shell lines run in
the run directory before `run_command`) and `save_time_limit` ("00:30:00").
Those values fill the `string.Template` `$name` placeholders in `templates/slurm_array.sh`
and `templates/slurm_save.sh`. `run.dry_run: true` writes inputs and scripts
and returns synthetic job IDs instead of submitting.

### Running on Pitagora

The save job runs `python -m gyrorun.save` with `sys.executable` of the
process that ran `gyrorun`, so it reads output with the same interpreter, and
the same `uv.lock`-pinned pyrokinetics commit, that wrote the inputs. There is
no second environment to configure:

```bash
cd GyroRun && uv sync
uv run gyrorun config.yaml
```

### Post-processing

`postprocess.py` reads generated GENE `parameters`, `omega`, and `nrg` output
through Pyrokinetics and f90nml. It identifies passive species, derives
transport coefficients by least-squares fitting, and combines per-directory
analysis results in a pandas DataFrame.

## 4. Dependencies and External Systems

- Python 3.11+ package managed with uv (pyrokinetics caps numpy at 2.1, which
  has no Python 3.14 wheel).
- Pyrokinetics, pinned to `feature/claude/open-prs-combined`, writes all inputs,
  builds scans and cubes, and reads all outputs.
- SciPy draws the Latin hypercube samples.
- The simulation code is supplied through `slurm.run_command`.
- Slurm (`sbatch`) runs the simulations; the cluster partition, account and
  time limits are configured in `config.yaml`.

No web service, database, authentication layer, or CI configuration is
present in this repository.

## 5. Development and Testing

Tests use pytest and cover parallelisation validation, Slurm rendering,
post-processing, dry-run scan and cube building, and the netCDF save. They use
only pyrokinetics' public template inputs and outputs. Run them with:

```bash
uv run pytest -q
```

The configured executable and Slurm settings are environment-specific. Use
`run.dry_run: true` when exercising scan generation without a cluster
submission.
