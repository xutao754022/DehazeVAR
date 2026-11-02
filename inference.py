#!/usr/bin/env python3
"""Directory-to-directory dehazing inference entry point for DehazeVAR."""

from __future__ import annotations

import argparse

from runners.evaluator import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_argument_group("inference inputs")
    input_group.add_argument("--config", required=True)
    input_group.add_argument("--checkpoint", required=True)
    input_group.add_argument("--input-dir", required=True)
    input_group.add_argument("--output-dir", required=True)

    runtime_group = parser.add_argument_group("runtime controls")
    runtime_group.add_argument(
        "--target-domain",
        choices=["clean", "hazy"],
        default="clean",
    )
    runtime_group.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    # Stage 1: parse inference paths and runtime options.
    args = parse_args()
    # Stage 2: load DehazeVAR checkpoint and export restored PNG predictions.
    run_inference(args)


if __name__ == "__main__":
    main()
