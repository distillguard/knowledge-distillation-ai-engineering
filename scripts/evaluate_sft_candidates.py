"""Run shortlisted SFT generation and optional classification with the shared GPU lock."""
import argparse
import sys

from filelock import FileLock
import evaluate_generation_benchmarks as evaluation
from dolly_scope import MODELS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", choices=MODELS, required=True)
    parser.add_argument("--classification", action="store_true")
    args = parser.parse_args(argv)
    with FileLock(str(evaluation.legacy.LOCAL / "worker.lock"), timeout=0):
        for name in args.model:
            evaluation.evaluate(name)
            if args.classification:
                previous = sys.argv
                try:
                    sys.argv = [evaluation.legacy.__file__, "--model", name]
                    evaluation.legacy.main()
                finally:
                    sys.argv = previous


if __name__ == "__main__":
    main()
