"""Run the already verified Dolly training, evaluation and report sequentially."""

from datetime import datetime, timezone
import os
import subprocess
import sys
from filelock import FileLock
from dolly_sft import LOCAL, OUT, ROOT, protocol, write_json


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    with FileLock(str(LOCAL / "worker.lock"), timeout=0):
        identity = {"protocol_fingerprint": protocol()["fingerprint"], "pid": os.getpid(),
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "log": (LOCAL / "worker.log").relative_to(ROOT).as_posix()}
        stage = "starting"
        try:
            with (LOCAL / "worker.log").open("a", encoding="utf-8") as log:
                for stage, arguments in (("training", ["dolly_sft.py", "train"]),
                                         ("evaluation", ["evaluate_dolly_sft.py"]),
                                         ("report", ["report_dolly_sft.py"])):
                    write_json(OUT / "worker_status.json", {**identity, "status": "running", "stage": stage})
                    log.write(f"\n{datetime.now(timezone.utc).isoformat()} {stage}\n"); log.flush()
                    subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / "scripts" / arguments[0]),
                                    *arguments[1:]], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            write_json(OUT / "worker_status.json", {**identity, "status": "completed", "stage": "completed",
                                                   "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        except Exception as error:
            write_json(OUT / "worker_status.json", {**identity, "status": "failed", "stage": stage,
                                                   "error": str(error), "failed_at_utc": datetime.now(timezone.utc).isoformat()})
            raise


if __name__ == "__main__":
    main()
