"""Keep restored-checkpoint evaluation out of epoch learning curves."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_dolly_sft import epoch_losses


class DevelopmentCurveTests(unittest.TestCase):
    def test_final_best_checkpoint_is_not_a_new_epoch_measurement(self):
        history = [{"epoch": 1, "loss": 2.5},
                   {"epoch": 1, "eval_loss": 2.0},
                   {"epoch": 2, "eval_loss": 1.0},
                   {"epoch": 3, "eval_loss": 1.5},
                   {"epoch": 3, "eval_loss": 1.0}]
        self.assertEqual(epoch_losses(history), {1: 2.0, 2: 1.0, 3: 1.5})


if __name__ == "__main__":
    unittest.main()
