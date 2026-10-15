"""Check scheduling and the shared lock without starting training or touching study outputs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_dolly_sft as worker
from filelock import FileLock, Timeout


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=worker.ROOT / "tmp")
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.local, self.out = root / "worker", root / "results"
        for name, value in (("LOCAL", self.local), ("OUT", self.out),
                            ("protocol", lambda: {"fingerprint": "test"})):
            patcher = patch.object(worker, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def status(self):
        return json.loads((self.out / "worker_status.json").read_text(encoding="utf-8"))

    def test_one_model_uses_correct_engine(self):
        for name, engine in (("gpt2_base", "dolly_sft.py"), ("opt_27b", "dolly_study.py")):
            with self.subTest(model=name), patch.object(worker.subprocess, "run") as run:
                worker.main(["--model", name])
                commands = [call.args[0] for call in run.call_args_list]
                self.assertEqual(len(commands), 4)
                self.assertEqual([Path(command[5]).name for command in commands],
                                 [engine, "report_dolly_sft.py", "evaluate_dolly_sft.py", "report_dolly_sft.py"])
                self.assertEqual(commands[0][-3:], ["train", "--model", name])
                self.assertEqual(commands[2][-2:], ["--model", name])
                self.assertEqual(self.status()["requested_models"], [name])
                self.assertEqual(self.status()["study_models"], list(worker.MODELS))
                self.assertEqual(self.status()["status"], "completed")

    def test_default_runs_all_models_in_order(self):
        with patch.object(worker.subprocess, "run") as run:
            worker.main([])
            self.assertEqual(run.call_count, 4 * len(worker.MODELS))
            self.assertEqual([call.args[0][-1] for call in run.call_args_list[::4]], list(worker.MODELS))

    def test_held_lock_prevents_duplicate_execution(self):
        self.local.mkdir()
        with FileLock(str(self.local / "worker.lock")), patch.object(worker.subprocess, "run") as run:
            with self.assertRaises(Timeout):
                worker.main(["--model", "gpt2_base"])
            run.assert_not_called()
            self.assertFalse((self.out / "worker_status.json").exists())

    def test_failure_stops_before_next_model(self):
        with patch.object(worker.subprocess, "run", side_effect=[None, None, subprocess.CalledProcessError(1, "test")]) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                worker.main([])
            self.assertEqual(run.call_count, 3)
            self.assertEqual(self.status()["status"], "failed")
            self.assertEqual(self.status()["stage"], "evaluation")
            self.assertEqual(self.status()["model"], worker.MODELS[0])


if __name__ == "__main__":
    unittest.main()
