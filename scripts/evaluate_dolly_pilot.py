"""Run only Dolly using the existing, fingerprinted generation implementation."""
import argparse

import evaluate_generation_benchmarks as evaluation
from dolly_scope import MODELS, REQUIRED_BENCHMARKS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    name = parser.parse_args(argv).model
    original = evaluation.BENCHMARKS
    try:
        # Restrict scheduling without changing generation, scoring or existing result identities.
        evaluation.BENCHMARKS = {key: original[key] for key in REQUIRED_BENCHMARKS}
        evaluation.evaluate(name)
    finally:
        evaluation.BENCHMARKS = original


if __name__ == "__main__":
    main()
