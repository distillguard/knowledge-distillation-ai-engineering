from pathlib import Path
import sys
import unittest
import tempfile
import os
import threading
import time
import importlib.util
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import monitor_ecqa_gpu as monitor


class MonitorTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows sharing semantics")
    def test_status_retry_recovers_a_real_temporary_windows_reader_lock(self):
        spec = importlib.util.spec_from_file_location("status_recovery", monitor.LOCAL / "resume_status_io.py")
        recovery = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recovery)
        folder = monitor.ROOT / "tmp"
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            path = Path(temp) / "worker_status.json"
            path.write_text('{"state": "before"}', encoding="utf-8")
            replacement = Path(temp) / "worker_status.json.tmp"
            replacement.write_text('{"state": "after"}', encoding="utf-8")
            reader = path.open("r", encoding="utf-8")
            with self.assertRaises(PermissionError):
                recovery.ORIGINAL_REPLACE(replacement, path)
            closer = threading.Timer(.2, reader.close)
            closer.start()
            try:
                recovery.status_replace(replacement, path)
                self.assertEqual(monitor.read(path)["state"], "after")
            finally:
                closer.join()
                reader.close()
            with patch.object(recovery, "ORIGINAL_REPLACE", side_effect=PermissionError), patch.object(recovery.time, "sleep") as sleep:
                with self.assertRaises(PermissionError):
                    recovery.status_replace(Path(temp)/"adapter.safetensors.tmp", Path(temp)/"adapter.safetensors")
                sleep.assert_not_called()
                with self.assertRaises(PermissionError):
                    recovery.status_replace(replacement, path)
                self.assertEqual(sleep.call_count, 40)

    def test_alert_boundaries_match_worker_limits(self):
        vram, ram = monitor.LIMITS["minimum_free_vram_bytes"], monitor.LIMITS["minimum_available_ram_bytes"]
        self.assertEqual(monitor.classify([{"free_bytes": vram}], ram), [])
        self.assertEqual(monitor.classify([{"free_bytes": vram-1}], ram), ["free_vram_below_desktop_reserve"])
        self.assertEqual(monitor.classify([{"free_bytes": vram}], ram-1), ["available_system_ram_below_reserve"])

    def test_minima_and_hourly_buckets_survive_restarts(self):
        summary = {}
        for utc, free in (("2026-10-27T00:59:59+00:00", 3), ("2026-10-27T01:00:14+00:00", 2)):
            item = {"utc": utc, "gpus": [{"free_bytes": free, "used_bytes": 10-free, "temperature_c": 55}],
                "available_ram_bytes": 10, "alerts": ["test"] if free == 2 else []}
            monitor.update_summary(summary, item)
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["minimum_free_vram_bytes"], 2)
        self.assertEqual(summary["critical_samples"], 1)
        self.assertEqual(len(summary["by_utc_hour"]), 2)
        self.assertEqual(summary["by_utc_hour"]["2026-10-27T00"]["minimum_free_vram_bytes"], 3)


if __name__ == "__main__":
    unittest.main()
