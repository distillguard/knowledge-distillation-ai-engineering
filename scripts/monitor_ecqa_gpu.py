"""Read-only 15-second GPU/RAM telemetry beside ECQA v2; never controls training."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import subprocess
import time

from filelock import FileLock
import psutil
from gpu_safety import LIMITS

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "local/ecqa_kd_v2"
INTERVAL = 15
MIB = 1024**2


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, data):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def classify(gpus, available_ram):
    alerts = []
    if any(g["free_bytes"] < LIMITS["minimum_free_vram_bytes"] for g in gpus):
        alerts.append("free_vram_below_desktop_reserve")
    if available_ram < LIMITS["minimum_available_ram_bytes"]:
        alerts.append("available_system_ram_below_reserve")
    return alerts


def sample():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    gpus = []
    for row in csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True):
        idx, name, total, used, free, utilization, temperature, power = row
        gpus.append({"index": int(idx), "name": name, "total_bytes": int(total)*MIB,
            "used_bytes": int(used)*MIB, "free_bytes": int(free)*MIB,
            "utilization_percent": float(utilization), "temperature_c": float(temperature),
            "power_watts": None if power in ("[N/A]", "N/A") else float(power)})
    assert gpus and all(0 <= g["free_bytes"] <= g["total_bytes"] for g in gpus)
    worker_path = LOCAL / "worker_status.json"
    worker = read(worker_path) if worker_path.exists() else {}
    available_ram = psutil.virtual_memory().available
    return {"utc": datetime.now(timezone.utc).isoformat(), "monitor_pid": os.getpid(), "gpus": gpus,
        "available_ram_bytes": available_ram, "worker": {k: worker[k] for k in
            ("pid", "stage", "seed", "experiment", "global_step", "completed", "total", "updated_utc") if k in worker},
        "alerts": classify(gpus, available_ram), "source": "nvidia-smi MiB; psutil RAM", "interval_seconds": INTERVAL}


def update_summary(summary, item):
    summary.setdefault("first_utc", item["utc"])
    summary["updated_utc"] = item["utc"]
    summary["interval_seconds"] = INTERVAL
    summary["limits"] = LIMITS
    # Sampled minima are not the true between-sample peak; the in-worker guards remain authoritative.
    for bucket in (summary, summary.setdefault("by_utc_hour", {}).setdefault(item["utc"][:13], {})):
        bucket["samples"] = bucket.get("samples", 0)+1
        bucket["critical_samples"] = bucket.get("critical_samples", 0)+bool(item["alerts"])
        for key, value, fn in (("minimum_free_vram_bytes", min(g["free_bytes"] for g in item["gpus"]), min),
                ("minimum_available_ram_bytes", item["available_ram_bytes"], min),
                ("maximum_used_vram_bytes", max(g["used_bytes"] for g in item["gpus"]), max),
                ("maximum_temperature_c", max(g["temperature_c"] for g in item["gpus"]), max)):
            bucket[key] = fn(bucket.get(key, value), value)
    if item["alerts"]:
        summary["last_critical_sample"] = item
    return summary


def run(once=False):
    LOCAL.mkdir(parents=True, exist_ok=True)
    with FileLock(str(LOCAL / "gpu_monitor.lock"), timeout=0):
        summary_path = LOCAL / "gpu_monitor_summary.json"
        summary = read(summary_path) if summary_path.exists() else {}
        while not (LOCAL / "STOP_GPU_MONITOR").exists():
            started = time.monotonic()
            try:
                item = sample()
                with (LOCAL / "gpu_monitor_samples.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+"\n")
                write(LOCAL / "gpu_monitor_latest.json", item)
                write(summary_path, update_summary(summary, item))
            except Exception as error:
                write(LOCAL / "gpu_monitor_error.json", {"utc": datetime.now(timezone.utc).isoformat(),
                    "monitor_pid": os.getpid(), "type": type(error).__name__, "reason": str(error)})
                if once:
                    raise
            if once:
                return
            time.sleep(max(1, INTERVAL-(time.monotonic()-started)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    run(parser.parse_args().once)
