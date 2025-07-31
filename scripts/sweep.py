#!/usr/bin/env python3
"""
scripts/sweep.py

Hyperparameter sweep utility: iterates over all YAML config files in a directory
and launches training runs sequentially. Logs output to separate files.

python scripts/sweep.py --config_dir configs --device cuda:0,1 --epochs 300 --resume --parallel 7

python scripts/sweep.py --config_dir configs --device cuda:0,1 --epochs 300 --resume --parallel 2


Hyperparameter sweep utility: iterates over all YAML config files in a directory
and launches training runs in parallel (up to --parallel). Logs output to separate files.

Usage:
  python -m scripts.sweep \
    --config_dir configs \
    --device cuda:0 \
    --parallel 2 \
    [--resume] \
    [--epochs 300]
"""

import os
import glob
import argparse
import subprocess
import time
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep over YAML configs"
    )
    parser.add_argument(
        "--config_dir", type=str, required=True,
        help="Directory containing YAML config files to sweep"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to pass to training runs (e.g., 'cuda:0')"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="If set, pass --resume to training calls for existing checkpoints"
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of parallel processes to run (default: 1)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs for all runs (e.g. 300)"
    )
    return parser.parse_args()


def launch_process(cmd, log_path):
    """
    Launch a subprocess with stdout/stderr redirected to log file.
    Returns the Popen object.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, 'w')
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)


def main():
    args = parse_args()

    # Find all YAML configs
    pattern = os.path.join(args.config_dir, "*.yaml")
    configs = sorted(glob.glob(pattern))
    if not configs:
        print(f"No YAML config files found in {args.config_dir}")
        sys.exit(1)

    print(f"Found {len(configs)} configuration(s) to run:")
    for cfg in configs:
        print(f"  - {os.path.basename(cfg)}")

    processes = []
    try:
        for cfg_path in configs:
            base = os.path.splitext(os.path.basename(cfg_path))[0]

            # Build command: run training as a module
            cmd = [
                sys.executable, "-m", "scripts.train",
                "--config", cfg_path,
                "--device", args.device
            ]
            if args.resume:
                cmd.append("--resume")
            if args.epochs is not None:
                cmd += ["--epochs", str(args.epochs)]

            log_path = os.path.join("logs", f"{base}.log")
            print(f"\nLaunching: {' '.join(cmd)}\n  -> log: {log_path}")

            proc = launch_process(cmd, log_path)
            processes.append(proc)

            # If at capacity, wait for one to finish
            while len(processes) >= args.parallel:
                time.sleep(5)
                processes = [p for p in processes if p.poll() is None]

        # Wait for remaining
        for p in processes:
            p.wait()

    except KeyboardInterrupt:
        print("\nSweep interrupted. Terminating running processes...")
        for p in processes:
            p.terminate()
        print("Cleanup complete.")
        sys.exit(1)

    print("\nSweep complete.")

if __name__ == "__main__":
    main()
