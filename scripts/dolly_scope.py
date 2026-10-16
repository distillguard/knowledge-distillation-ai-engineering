"""Active pilot scope; preserve the fingerprinted preparation and excluded Qwen results."""
from dolly_study import MODELS as PREPARED_MODELS


EXCLUDED_MODELS = ("qwen_08b", "qwen_9b")
MODELS = tuple(name for name in PREPARED_MODELS if name not in EXCLUDED_MODELS)
REQUIRED_BENCHMARKS = ("dolly",)
