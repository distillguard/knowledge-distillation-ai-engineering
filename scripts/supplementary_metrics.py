"""Saved-response metric helpers; no model loading or training imports."""
import math
from pathlib import Path
import statistics
import sys

# Keep the environment's compiled packages ahead of the existing pure-Python evaluation dependencies.
sys.path.append(str(Path(__file__).resolve().parents[1] / "tmp/eval_deps"))
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.translate.meteor_score import meteor_score


def word_tokens(text):
    return word_tokenize(text, preserve_line=True)


def meteor(response, references):
    assert references and all(isinstance(reference, str) and reference.strip() for reference in references)
    return meteor_score([word_tokens(reference) for reference in references], word_tokens(response), wordnet=wordnet)


def distinct4(responses):
    unique, total = set(), 0
    for response in responses:
        tokens = word_tokens(response)
        grams = list(zip(tokens, tokens[1:], tokens[2:], tokens[3:]))
        unique.update(grams)
        total += len(grams)
    return {"unique_4grams": len(unique), "total_4grams": total,
            "distinct4": len(unique) / total if total else None}


def calibration(labels, predictions, positive_probabilities):
    assert labels and len(labels) == len(predictions) == len(positive_probabilities)
    assert all(label in (0, 1) for label in labels) and all(prediction in (0, 1) for prediction in predictions)
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in positive_probabilities)
    confidence = [max(p, 1 - p) for p in positive_probabilities]
    correct = [int(label == prediction) for label, prediction in zip(labels, predictions)]
    bins = []
    # Use Guo et al.'s released default: 15 equal-width (lower, upper] bins.
    for index in range(15):
        lower, upper = index / 15, (index + 1) / 15
        members = [i for i, value in enumerate(confidence) if lower < value <= upper]
        accuracy = statistics.mean(correct[i] for i in members) if members else None
        average = statistics.mean(confidence[i] for i in members) if members else None
        contribution = len(members) / len(labels) * abs(accuracy - average) if members else 0
        bins.append({"lower_exclusive": lower, "upper_inclusive": upper, "n": len(members),
                     "accuracy": accuracy, "mean_confidence": average, "ece_contribution": contribution})
    assert sum(row["n"] for row in bins) == len(labels)
    return {"n": len(labels), "ece": math.fsum(row["ece_contribution"] for row in bins), "bins": bins}
