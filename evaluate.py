#!/usr/bin/env python3
"""Multi-dataset evaluation entry point for DehazeVAR."""

from __future__ import annotations

import argparse

from runners.evaluator import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_argument_group("evaluation inputs")
    input_group.add_argument("--config", required=True)
    input_group.add_argument("--checkpoint", required=True)
    input_group.add_argument(
        "--output-directory",
        "--output-dir",
        "--output",
        dest="output_directory",
        help="Output root directory; defaults to <run>/evaluations.",
    )
    input_group.add_argument(
        "--dataset-names",
        "--test-sets",
        dest="dataset_names",
        nargs="+",
        help=(
            "Explicit evaluation dataset names. Defaults to "
            "evaluation.production_test_sets from the config."
        ),
    )

    runtime_group = parser.add_argument_group("runtime controls")
    runtime_group.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional subset size for quick verification (0 evaluates all images).",
    )
    runtime_group.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
