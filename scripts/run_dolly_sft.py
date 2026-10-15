"""Train, document and evaluate one model at a time across five model families."""

import argparse
from datetime import datetime, timezone
import os
import subprocess
import sys
from filelock import FileLock
from dolly_sft import LOCAL, OUT, ROOT, protocol, write_json
from dolly_study import ADDED, MODELS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    args = parser.parse_args(argv)
    names = MODELS if args.model == "all" else (args.model,)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with FileLock(str(LOCAL / "worker.lock"), timeout=0):
        identity = {"protocol_fingerprint": protocol()["fingerprint"], "pid": os.getpid(),
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "log": (LOCAL / "worker.log").relative_to(ROOT).as_posix(),
                    "study_models": list(MODELS), "requested_models": list(names)}
        stage = "starting"
        name = None
        try:
            with (LOCAL / "worker.log").open("a", encoding="utf-8") as log:
                for name in names:
                    commands = (("training", ["dolly_study.py" if name in ADDED else "dolly_sft.py", "train", "--model", name]),
                                ("sft_documentation", ["report_dolly_sft.py"]),
                                ("evaluation", ["evaluate_dolly_sft.py", "--model", name]),
                                ("report", ["report_dolly_sft.py"]))
                    for stage, arguments in commands:
                        write_json(OUT / "worker_status.json", {**identity, "status": "running", "stage": stage, "model": name})
                        log.write(f"\n{datetime.now(timezone.utc).isoformat()} {stage} {name}\n"); log.flush()
                        subprocess.run([sys.executable, "-u", "-B", "-X", "utf8", str(ROOT / "scripts" / arguments[0]),
                                        *arguments[1:]], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            write_json(OUT / "worker_status.json", {**identity, "status": "completed", "stage": "completed",
                                                   "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        except Exception as error:
            write_json(OUT / "worker_status.json", {**identity, "status": "failed", "stage": stage, "model": name,
                                                   "error": str(error), "failed_at_utc": datetime.now(timezone.utc).isoformat()})
            raise


if __name__ == "__main__":
    main()
