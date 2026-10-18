"""Keep restored-checkpoint evaluation out of epoch learning curves."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_dolly_sft import epoch_losses
import report_dolly_sft as report


class DevelopmentCurveTests(unittest.TestCase):
    def test_final_best_checkpoint_is_not_a_new_epoch_measurement(self):
        history = [{"epoch": 1, "loss": 2.5},
                   {"epoch": 1, "eval_loss": 2.0},
                   {"epoch": 2, "eval_loss": 1.0},
                   {"epoch": 3, "eval_loss": 1.5},
                   {"epoch": 3, "eval_loss": 1.0}]
        self.assertEqual(epoch_losses(history), {1: 2.0, 2: 1.0, 3: 1.5})

    def test_selected_sft_has_development_plot_before_auxiliary_evaluation(self):
        name = "gemma2_9b_it"
        with tempfile.TemporaryDirectory(dir=report.ROOT / "tmp") as folder:
            root = Path(folder)
            run = {"status": "completed", "protocol_fingerprint": "test",
                   "before_sft_development": {"before_sft_loss": 3},
                   "trainer_log_history": [{"epoch": 1, "eval_loss": 1.4},
                                           {"epoch": 2, "eval_loss": 1.5},
                                           {"epoch": 3, "eval_loss": 1.6},
                                           {"epoch": 3, "eval_loss": 1.4}]}
            for lr in report.LR_CANDIDATES:
                (root / f"{name}_lr{lr:.0e}.json").write_text(json.dumps(run), encoding="utf-8")
            spec = {"fingerprint": "test", "precision": {name: {"base_weights": "NF4 with double quantization"}}}
            with patch.object(report, "OUT", root), patch.object(report, "MODELS", (name,)), \
                    patch.object(report, "progress", return_value=[name]), \
                    patch.object(report, "result_dir", return_value=root), \
                    patch.object(report, "model_protocol", return_value=spec):
                report.main()
            self.assertTrue((root / "development_nll.png").exists())
            self.assertGreater((root / "development_nll.png").stat().st_size, 0)
            self.assertIn("development_nll.png", (root / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
