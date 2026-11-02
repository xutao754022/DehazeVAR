#!/usr/bin/env python3
"""Stage-2 joint unpaired dehazing training entry point for DehazeVAR."""

from __future__ import annotations

import argparse
from pathlib import Path

from runners.dehazevar_trainer import run_dehazevar_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_argument_group("experiment inputs")
    input_group.add_argument("--config", required=True)
    input_group.add_argument("--run-dir", "--output-dir", dest="run_dir", required=True)
    input_group.add_argument("--resume")
    input_group.add_argument(
        "--prt-checkpoint",
        "--disen-vae-checkpoint",
        dest="prt_checkpoint",
        type=Path,
        help="Optional override for model.disen_vae_pretrained_checkpoint.",
    )

    training_group = parser.add_argument_group("training overrides")
    training_group.add_argument("--max-steps", type=int)
    training_group.add_argument("--batch-size", type=int)
    training_group.add_argument("--checkpoint-every", type=int)
    training_group.add_argument("--sample-every", type=int)
    training_group.add_argument("--num-workers", type=int)
    training_group.add_argument(
        "--adapter-pretraining-steps",
        type=int,
        help="Override the adapter-only pretraining length before DehazeVAR step zero.",
    )
    training_group.add_argument(
        "--adapter-identity-pretraining-steps",
        type=int,
        help="Override clear identity-branch adapter pretraining before joint step zero.",
    )
    training_group.add_argument("--allow-non-production-prt", action="store_true")
    training_group.add_argument("--overfit-single-pair", action="store_true")
    training_group.add_argument("--skip-initial-checkpoint", action="store_true")
    training_group.add_argument("--skip-final-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    # Stage 1: parse command-line arguments and overrides.
    args = parse_args()
    # Stage 2: run CRA warm-up and joint bidirectional DehazeVAR training.
    run_dehazevar_training(args)


if __name__ == "__main__":
    main()
