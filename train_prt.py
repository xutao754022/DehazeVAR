#!/usr/bin/env python3
"""Stage-1 pretraining entry point for the Parallax-Resilient Tokenizer (PRT)."""

from __future__ import annotations

import argparse
from pathlib import Path

from runners.prt_trainer import run_prt_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_argument_group("experiment inputs")
    input_group.add_argument("--config", required=True)
    input_group.add_argument(
        "--output-dir",
        "--run-directory",
        "--run-dir",
        dest="run_directory",
        type=Path,
        required=True,
    )
    input_group.add_argument("--resume", type=Path)

    training_group = parser.add_argument_group("training overrides")
    training_group.add_argument("--max-steps", type=int)
    training_group.add_argument("--heldout-clean-count", type=int)
    training_group.add_argument(
        "--workers", "--num-workers", dest="workers", type=int
    )
    training_group.add_argument("--skip-initial-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_prt_training(args)


if __name__ == "__main__":
    main()
