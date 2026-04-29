#!/usr/bin/env python3
"""
Run a game-theoretic prompting experiment from a YAML config.

Usage:
    python scripts/run_experiment.py experiments/pilot.yaml
    python scripts/run_experiment.py experiments/pilot.yaml --output results/
    python scripts/run_experiment.py experiments/pilot.yaml --dry-run

    # Resume a partially failed experiment — only re-runs missing/errored runs.
    python scripts/run_experiment.py experiments/pilot.yaml \\
        --resume results/pilot_20260423_053833
"""
import argparse
import sys
import os

# Add project root to path so gtp package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from gtp.config import load_experiment
from gtp.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run game-theoretic prompting experiments"
    )
    parser.add_argument("config", help="Path to YAML experiment config")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Resume an existing results directory instead of creating a new "
            "one. Skips runs whose runs/<id>.json already exists; only "
            "missing/previously-errored runs are re-executed."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show expanded grid without running",
    )
    args = parser.parse_args()

    config = load_experiment(args.config)
    runner = ExperimentRunner(
        config, output_dir=args.output, resume_path=args.resume
    )

    if args.dry_run:
        runner.dry_run()
        return

    results_path = runner.run(config_path=args.config)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
