"""Compare chunked CE with the installed pinned Transformers loss on CPU."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from transformers.loss.loss_utils import ForCausalLMLoss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dolly_chunked_loss as chunked


class ChunkedLossTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator(device="cpu").manual_seed(17)
        self.vocab_size = 37
        self.tokens = chunked.CHUNK_TOKENS * 2 + 7

    def examples(self, dtype=torch.float32):
        logits = torch.randn(2, self.tokens, self.vocab_size, generator=self.generator, dtype=dtype)
        labels = torch.randint(self.vocab_size, (2, self.tokens), generator=self.generator)
        # The first complete chunk has no targets, the response begins exactly
        # at a shifted chunk boundary, and the last response token is EOS.
        labels[0, :chunked.CHUNK_TOKENS + 1] = -100
        labels[0, -6] = self.vocab_size - 1
        labels[0, -5:] = -100
        # A second example starts its response across a different chunk boundary.
        labels[1, :chunked.CHUNK_TOKENS - 2] = -100
        labels[1, -1] = self.vocab_size - 1
        return logits, labels

    def compare(self, logits, labels, **kwargs):
        expected_logits = logits.clone().requires_grad_(True)
        actual_logits = logits.clone().requires_grad_(True)
        expected = ForCausalLMLoss(expected_logits, labels, self.vocab_size, **kwargs)
        actual = chunked.chunked_causal_lm_loss(actual_logits, labels, self.vocab_size, **kwargs)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
        expected.backward()
        actual.backward()
        self.assertTrue(torch.isfinite(actual_logits.grad).all())
        torch.testing.assert_close(actual_logits.grad, expected_logits.grad, rtol=3e-6, atol=2e-7)
        return actual_logits.grad

    def test_fp32_mask_shift_eos_and_partial_chunk(self):
        logits, labels = self.examples()
        grad = self.compare(logits, labels)
        shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
        self.assertTrue(torch.equal(grad[shifted == -100], torch.zeros_like(grad[shifted == -100])))

    def test_integer_and_tensor_accumulation_denominators(self):
        logits, labels = self.examples()
        count = int(labels[..., 1:].ne(-100).sum()) * 3
        for denominator in (count, torch.tensor(count)):
            with self.subTest(denominator=type(denominator).__name__):
                self.compare(logits, labels, num_items_in_batch=denominator)

    def test_provided_shift_labels_are_not_shifted_again(self):
        logits, labels = self.examples()
        shifted = torch.randint(self.vocab_size, labels.shape, generator=self.generator)
        shifted[:, :chunked.CHUNK_TOKENS] = -100
        shifted[:, -1] = self.vocab_size - 1
        self.compare(logits, labels, shift_labels=shifted)

    def test_custom_ignore_index(self):
        logits, labels = self.examples()
        labels[labels == -100] = -7
        self.compare(logits, labels, ignore_index=-7)

    def test_bf16_inputs_keep_fp32_loss_and_match_reference_gradients(self):
        logits, labels = self.examples()
        self.compare(logits.to(torch.bfloat16), labels)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA loss conformance requires a GPU")
    def test_cuda_bf16_gemma_vocabulary_matches_reference(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 GPU is required")
        self.vocab_size = 256000
        logits, labels = self.examples()
        self.compare(logits.to(device="cuda", dtype=torch.bfloat16), labels.to("cuda"))

    def test_accumulated_microbatches_use_one_shared_token_denominator(self):
        logits, labels = self.examples()
        expected_logits = logits.clone().requires_grad_(True)
        actual_logits = logits.clone().requires_grad_(True)
        denominator = labels[..., 1:].ne(-100).sum()
        expected = ForCausalLMLoss(expected_logits, labels, self.vocab_size, num_items_in_batch=denominator)
        actual = sum(chunked.chunked_causal_lm_loss(
            actual_logits[index:index + 1], labels[index:index + 1], self.vocab_size,
            num_items_in_batch=denominator,
        ) for index in range(logits.shape[0]))
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
        expected.backward()
        actual.backward()
        torch.testing.assert_close(actual_logits.grad, expected_logits.grad, rtol=3e-6, atol=2e-7)

    def test_eval_loss_matches_without_checkpointing(self):
        logits, labels = self.examples()
        with torch.no_grad(), patch.object(chunked, "checkpoint", side_effect=AssertionError("eval checkpoint")):
            actual = chunked.chunked_causal_lm_loss(logits, labels, self.vocab_size)
            expected = ForCausalLMLoss(logits, labels, self.vocab_size)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)

    def test_fp32_conversion_and_ce_are_bounded_by_chunk_size(self):
        logits, labels = self.examples()
        logits = logits.to(torch.bfloat16).requires_grad_(True)
        calls = []
        reference_ce = torch.nn.functional.cross_entropy
        def record(source, target, **kwargs):
            calls.append((source.shape, source.dtype))
            return reference_ce(source, target, **kwargs)
        with patch.object(chunked.F, "cross_entropy", side_effect=record):
            chunked.chunked_causal_lm_loss(logits, labels, self.vocab_size).backward()
        self.assertGreater(len(calls), 2)
        self.assertTrue(all(shape[0] <= chunked.CHUNK_TOKENS and dtype == torch.float32 for shape, dtype in calls))

    def test_fully_ignored_or_empty_mean_matches_nan_and_zero_reference_gradient(self):
        for length in (self.tokens, 0):
            with self.subTest(length=length):
                logits = torch.randn(1, length, self.vocab_size, generator=self.generator)
                expected_logits = logits.clone().requires_grad_(True)
                actual_logits = logits.clone().requires_grad_(True)
                labels = torch.full((1, length), -100, dtype=torch.long)
                expected = ForCausalLMLoss(expected_logits, labels, self.vocab_size)
                actual = chunked.chunked_causal_lm_loss(actual_logits, labels, self.vocab_size)
                self.assertTrue(torch.isnan(expected))
                self.assertTrue(torch.isnan(actual))
                expected.backward()
                actual.backward()
                torch.testing.assert_close(actual_logits.grad, expected_logits.grad, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
