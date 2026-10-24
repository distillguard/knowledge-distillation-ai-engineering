import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ecqa_study as study
import gpu_safety
import sft_ecqa
from render_results import update_section


class EcqaTests(unittest.TestCase):
    def test_split_removes_id_and_normalized_question_leakage(self):
        rows = {"a": {"id": "a", "question": "Hello  WORLD"},
                "b": {"id": "b", "question": "hello world"},
                "c": {"id": "c", "question": "Different"}}
        result, removed = study.clean_split(rows, {"test": ["a"], "dev": ["b"], "train": ["a", "b", "c"]})
        self.assertEqual([r["id"] for r in result["test"]], ["a"])
        self.assertEqual(result["dev"], [])
        self.assertEqual([r["id"] for r in result["train"]], ["c"])
        self.assertEqual(removed["dev"], ["b"])

    def test_prompt_never_contains_gold_or_explanation(self):
        row = {"question": "QUESTION", "options": {c: "OPTION_"+c for c in "ABCDE"},
               "gold": "GOLD_SECRET", "explanation": "EXPLANATION_SECRET", "positives": ["POSITIVE_SECRET"]}
        prompt = study.prompt(row, 256)
        for secret in ("GOLD_SECRET", "EXPLANATION_SECRET", "POSITIVE_SECRET"):
            self.assertNotIn(secret, prompt)
        self.assertEqual(prompt.count("OPTION_"), 5)

    def test_parser_does_not_find_an_answer_inside_explanation(self):
        self.assertIsNone(study.inspect_response("EXPLANATION: ANSWER: E")["predicted"])
        self.assertIsNone(study.inspect_response("ANSWER: A or E\nEXPLANATION: text")["predicted"])
        result = study.inspect_response("ANSWER: E\nEXPLANATION: unfinished")
        self.assertEqual(result["predicted"], "E")
        self.assertTrue(result["format_valid"])
        self.assertFalse(study.inspect_response("ANSWER: E\nEXPLANATION:")["format_valid"])

    def test_macro_f1_and_cap_are_separate(self):
        rows = []
        for i, c in enumerate("ABCDE"):
            probabilities = [.05]*5
            probabilities[i] = .8
            rows.append({"gold": c, "predicted": c if c != "E" else None,
                "choice_probabilities": probabilities, "choice_log_probabilities": list(map(math.log, probabilities)),
                "hit_length_cap": c == "A", "explanation_present": True, "format_valid": c != "E",
                "output_tokens": 10, "seconds": 1})
        result = study.metrics(rows)
        self.assertEqual(result["accuracy"], .8)
        self.assertEqual(result["macro_f1"], .8)
        self.assertEqual(result["invalid_answer_count"], 1)
        self.assertEqual(result["length_cap_count"], 1)
        self.assertAlmostEqual(result["choice_nll"], -math.log(.8))
        self.assertAlmostEqual(result["brier_score"], .05)

    def test_ties_follow_token_id_not_label_order(self):
        import torch
        formatter = study.AnswerFirst(1, [0], [4, 6, 2, 5, 7], [1])
        scores = torch.tensor([[0., 0., 8., 0., 8., 0., 3., 0.]])
        result = formatter(torch.zeros((1, 2), dtype=torch.long), scores)
        self.assertEqual(int(result.argmax()), 2)
        self.assertEqual(formatter.selected_label, "C")
        self.assertAlmostEqual(sum(formatter.probabilities), 1., places=6)

    def test_safety_reserves_and_time_limit(self):
        g = gpu_safety.GIB
        self.assertEqual(gpu_safety.allocator_fraction(15*g, 16*g, 2*g), 13/16)
        with self.assertRaises(gpu_safety.ResourceLimitError):
            gpu_safety.allocator_fraction(2*g, 16*g, 2*g)
        self.assertIsNone(gpu_safety.violation(3*g, 8*g, 1))
        self.assertEqual(gpu_safety.violation(3*g, 3*g, 1), "available_system_ram_below_reserve")
        self.assertEqual(gpu_safety.violation(3*g, 8*g, 60), "question_time_limit")

    def test_report_section_preserves_surrounding_document(self):
        source = "before\n<!-- X_START -->\nold\n<!-- X_END -->\nafter"
        self.assertEqual(update_section(source, "X", "new"), source.replace("old", "new"))
        self.assertEqual(update_section(source, "MISSING", "new"), source)

    def test_dev_selection_does_not_use_test_and_breaks_ties_reproducibly(self):
        self.assertEqual(sft_ecqa.LEARNING_RATES, {"qwen35_4b": 5e-5, "qwen35_2b": 5e-5, "qwen35_08b": 1e-4})
        self.assertEqual(sft_ecqa.EPOCHS, 3)
        self.assertFalse(sft_ecqa.RECIPE["early_stopping"])
        for name, lr in sft_ecqa.LEARNING_RATES.items():
            candidates = [{"learning_rate": lr, "epoch": ep,
                "dev": {"accuracy": .5, "macro_f1": .4, "choice_nll": 1.}, "test_accuracy": 1.}
                for ep in (1, 2, 3)]
            selected = sft_ecqa.select_candidate(name, candidates)
            self.assertEqual((selected["learning_rate"], selected["epoch"]), (lr, 1))
            candidates[-1]["dev"]["accuracy"] = .6
            candidates[-1]["test_accuracy"] = 0.
            self.assertIs(sft_ecqa.select_candidate(name, candidates), candidates[-1])
            candidates[-1]["dev"]["accuracy"] = .5
            candidates[-1]["dev"]["macro_f1"] = .6
            self.assertIs(sft_ecqa.select_candidate(name, candidates), candidates[-1])
            candidates[-1]["dev"]["macro_f1"] = .4
            candidates[-1]["dev"]["choice_nll"] = .9
            self.assertIs(sft_ecqa.select_candidate(name, candidates), candidates[-1])
            with self.assertRaises(AssertionError):
                sft_ecqa.select_candidate(name, candidates[:2])
            candidates[-1]["learning_rate"] = lr*2
            with self.assertRaises(AssertionError):
                sft_ecqa.select_candidate(name, candidates)

    def test_response_ce_matches_full_causal_ce_with_prompt_mask(self):
        import torch
        from types import SimpleNamespace
        logits = torch.tensor([[[1., 2., 3., 4.], [2., 3., 1., 4.],
                                [3., 1., 2., 4.], [4., 3., 2., 1.]]])
        labels = torch.tensor([[-100, -100, 2, 1]])
        class Model:
            def __call__(self, **kwargs):
                return SimpleNamespace(logits=logits[:, -kwargs["logits_to_keep"]:])
        loss, _ = sft_ecqa.response_loss(Model(), {"input_ids": torch.tensor([[0, 1, 2, 3]]),
            "attention_mask": torch.ones((1, 4)), "labels": labels})
        expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 4), labels[:, 1:].reshape(-1))
        self.assertTrue(torch.equal(loss, expected))


if __name__ == "__main__":
    unittest.main()
