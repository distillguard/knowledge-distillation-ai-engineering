"""Run metric preparation/scoring in a child process without changing the SFT environment."""
import argparse
import os
import subprocess
import sys

from prepare_supplementary_metrics import ROOT, LOCAL


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score"))
    args, remaining = parser.parse_known_args(argv)
    if args.stage == "prepare" and remaining:
        parser.error("prepare takes no additional arguments")
    dependencies = LOCAL / "deps"
    assert dependencies.is_dir(), "Install requirements-supplementary-metrics.txt into the documented --target directory first"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(dependencies)
    environment["USE_TF"] = environment["USE_FLAX"] = "0"
    if args.stage == "score":
        environment["HF_HUB_OFFLINE"] = "1"
    script = "prepare_supplementary_metrics.py" if args.stage == "prepare" else "evaluate_supplementary_metrics.py"
    subprocess.run([sys.executable, "-u", "-B", "-X", "utf8", str(ROOT / "scripts" / script), *remaining],
                   cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
