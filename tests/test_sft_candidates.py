"""Protect candidate ordering, shared locking and legacy argument restoration."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_sft_candidates as candidate
from filelock import FileLock, Timeout


class CandidateTests(unittest.TestCase):
    def test_order_and_classification_argument_restoration_on_failure(self):
        calls = []
        previous = sys.argv
        def classify():
            calls.append(("classification", sys.argv[-1]))
            raise RuntimeError("classification interrupted")
        with tempfile.TemporaryDirectory(dir=candidate.evaluation.ROOT / "tmp") as folder, \
                patch.object(candidate.evaluation.legacy, "LOCAL", Path(folder)), \
                patch.object(candidate.evaluation, "evaluate", side_effect=lambda name: calls.append(("generation", name))), \
                patch.object(candidate.evaluation.legacy, "main", side_effect=classify):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                candidate.main(["--model", "gpt2_base", "opt_125m", "--classification"])
        self.assertIs(sys.argv, previous)
        self.assertEqual(calls, [("generation", "gpt2_base"), ("classification", "gpt2_base")])

    def test_generation_only_order_and_lock_before_inference(self):
        with tempfile.TemporaryDirectory(dir=candidate.evaluation.ROOT / "tmp") as folder, \
                patch.object(candidate.evaluation.legacy, "LOCAL", Path(folder)), \
                patch.object(candidate.evaluation, "evaluate") as run, \
                patch.object(candidate.evaluation.legacy, "main") as classify:
            with FileLock(str(Path(folder) / "worker.lock")):
                with self.assertRaises(Timeout):
                    candidate.main(["--model", "opt_125m"])
            run.assert_not_called()
            candidate.main(["--model", "gpt2_xl", "opt_125m"])
            self.assertEqual([call.args[0] for call in run.call_args_list], ["gpt2_xl", "opt_125m"])
            classify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
