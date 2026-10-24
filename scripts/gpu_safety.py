"""Resource guards for the fixed-512 rerun on the shared desktop GPU."""
import time

GIB = 1024 ** 3
LIMITS = {"minimum_free_vram_bytes": 2*GIB, "minimum_available_ram_bytes": 4*GIB,
          "maximum_seconds_per_question": 60}


class ResourceLimitError(RuntimeError):
    pass


def allocator_fraction(free, total, reserve):
    if not 0 < free <= total or free <= reserve:
        raise ResourceLimitError("Insufficient free VRAM before model load")
    return (free-reserve)/total


def violation(free_vram, available_ram, elapsed, limits=LIMITS):
    if free_vram < limits["minimum_free_vram_bytes"]:
        return "free_vram_below_desktop_reserve"
    if available_ram < limits["minimum_available_ram_bytes"]:
        return "available_system_ram_below_reserve"
    if elapsed >= limits["maximum_seconds_per_question"]:
        return "question_time_limit"
    return None


def configure():
    import torch
    free, total = torch.cuda.mem_get_info()
    fraction = allocator_fraction(free, total, LIMITS["minimum_free_vram_bytes"])
    torch.cuda.set_per_process_memory_fraction(fraction)
    return {"initial_free_bytes": free, "total_bytes": total, "allocator_fraction": fraction, **LIMITS}


def stopping_guard():
    import psutil
    import torch
    from transformers import StoppingCriteria

    class Guard(StoppingCriteria):
        def __init__(self):
            self.started = time.monotonic()
            self.reason = None
            self.minimum_free_vram = None

        def check(self):
            free, _ = torch.cuda.mem_get_info()
            self.minimum_free_vram = free if self.minimum_free_vram is None else min(free, self.minimum_free_vram)
            self.reason = violation(free, psutil.virtual_memory().available, time.monotonic()-self.started)
            if self.reason:
                raise ResourceLimitError(self.reason)

        def __call__(self, input_ids, scores, **kwargs):
            self.check()
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    guard = Guard()
    guard.check()
    return guard
