"""Compose run directories in the Gyrokinetic_Simulations layout:
<root>/<CODE>/Runs/<scan_name>/<case>/<leaf>"""
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SEGMENT = re.compile(r"[A-Za-z0-9._-]+")


def compose_leaf(code, run, leaf, root=None):
    """
    root is paths.output_dir_base when given (dry runs, tests), else
    $GYRO_DATA_OUTPUT. The tail is identical either way. Refuses a bad
    segment or a non-empty existing leaf: never overwrite a scan.
    """
    root = root or os.environ.get("GYRO_DATA_OUTPUT")
    if not root:
        raise SystemExit("GYRO_DATA_OUTPUT is not set and paths.output_dir_base is not given")
    parts = [code.upper(), "Runs", run["scan_name"], run["case"], *leaf.split("/")]
    for part in parts:
        if not SEGMENT.fullmatch(part) or part in (".", ".."):
            raise SystemExit(f"invalid directory segment {part!r} in {'/'.join(parts)}")
    path = Path(root, *parts)
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"refusing to write into existing non-empty directory {path}")
    return path


def kind(scan_config):
    return "cube" if "cube" in scan_config else "scan"


def write_manifest(leaf, config, scan_config, ids, executable=None):
    """Provenance for the run catalogue: what was built, by what, submitted when."""
    sha = subprocess.run(
        ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    exe = None
    if executable and Path(executable).is_file():
        st = Path(executable).stat()
        exe = {"path": str(executable), "size": st.st_size, "mtime": st.st_mtime}
    manifest = {
        "gyrorun_commit": sha,
        "config": {**config, "scans": [scan_config]},
        # convention the scan values are written in; None = plain floats
        "units": next((i["units"] for i in scan_config[kind(scan_config)] if "units" in i), None),
        "executable": exe,
        "submitted": datetime.now(timezone.utc).isoformat(),
        "array_job": ids["array"],
        "save_job": ids["save"],
        "dry_run": ids["dry_run"],
        "leaf": str(leaf),
    }
    (Path(leaf) / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
