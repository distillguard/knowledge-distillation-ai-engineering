"""Bound CausalLM CE workspace while retaining the pinned HF loss objective.

Gemma still produces the same BF16, soft-capped logits. Only FP32 conversion
and cross entropy are recomputed in token chunks. Reduction order can differ
from the full-matrix implementation, so this is not a bitwise-equivalence claim.
Only model workers import this module; the coordinator stays free of CUDA
contexts until its child GPU probes finish.
"""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


CHUNK_TOKENS = 64


def _cross_entropy_sum(logits, targets, ignore_index):
    # Keep the upcast inside the checkpoint. Upcasting before this call would
    # retain the full FP32 vocabulary matrix throughout the backward pass.
    return F.cross_entropy(logits.float(), targets, ignore_index=ignore_index, reduction="sum")


def chunked_causal_lm_loss(
    logits,
    labels,
    vocab_size,
    num_items_in_batch=None,
    ignore_index=-100,
    shift_labels=None,
    **kwargs,
):
    """Match HF ForCausalLMLoss masking, global shift and batch normalization.

    Trainer may pass the valid-token count across all accumulated microbatches.
    Each chunk therefore returns a sum; normalization happens exactly once,
    using that shared count when present, rather than averaging chunk means.
    """
    if shift_labels is None:
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1).to(logits.device)
    checkpoint_chunks = torch.is_grad_enabled() and logits.requires_grad
    loss = None
    for start in range(0, logits.shape[0], CHUNK_TOKENS):
        chunk = logits[start:start + CHUNK_TOKENS]
        targets = shift_labels[start:start + CHUNK_TOKENS]
        if checkpoint_chunks:
            partial = checkpoint(
                _cross_entropy_sum, chunk, targets, ignore_index,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            partial = _cross_entropy_sum(chunk, targets, ignore_index)
        loss = partial if loss is None else loss + partial

    if loss is None:
        # Preserve autograd for an empty input, as the reference CE does.
        loss = logits.sum().float()
    denominator = num_items_in_batch
    if denominator is None:
        denominator = shift_labels.ne(ignore_index).sum()
        # PyTorch's mean CE returns NaN with zero gradients when every target is
        # ignored (including empty input). Avoid a 0/0 backward in that case.
        mean = loss / denominator.clamp_min(1)
        return torch.where(denominator > 0, mean, mean + float("nan"))
    if torch.is_tensor(denominator):
        denominator = denominator.to(loss.device)
    return loss / denominator
