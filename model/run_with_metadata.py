"""Wrapper to run model/train.py while recording git commit and run metadata.

Usage:
    python model/run_with_metadata.py --mode world --epochs 50 --lr 1e-3

This will create a `model/runs/` directory with a timestamped subfolder containing
`metadata.json` and will invoke the training script with the same arguments.
"""

from pathlib import Path
import subprocess
import sys
import json
import argparse
import datetime

ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train.py"
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)


def git_commit_hash(repo_root: Path):
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
        return out.decode().strip()
    except Exception:
        return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser(description="Run train.py and record metadata.")
    parser.add_argument(
        "--mode", choices=["world", "flow", "diffusion"], default="world"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument(
        "--extra", nargs=argparse.REMAINDER, help="Extra args passed through"
    )
    args = parser.parse_args()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_folder = RUNS_DIR / f"run_{args.mode}_{timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)

    commit = git_commit_hash(ROOT.parent)

    # Build command
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--mode", args.mode]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.lr is not None:
        cmd += ["--lr", str(args.lr)]
    if args.latent_dim is not None:
        cmd += ["--latent-dim", str(args.latent_dim)]
    if args.timesteps is not None:
        cmd += ["--timesteps", str(args.timesteps)]
    if args.extra:
        cmd += args.extra

    metadata = {
        "timestamp_utc": timestamp,
        "git_commit": commit,
        "cmd": cmd,
    }

    meta_path = run_folder / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Starting training: {' '.join(cmd)}")
    print(f"Metadata written to {meta_path}")

    rc = subprocess.call(cmd, cwd=ROOT.parent)
    metadata["returncode"] = rc
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if rc != 0:
        print(f"Training exited with code {rc}")
        sys.exit(rc)


if __name__ == "__main__":
    main()
