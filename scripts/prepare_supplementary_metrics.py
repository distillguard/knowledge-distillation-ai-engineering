"""Prepare pinned English metric assets in the isolated scoring environment, without GPU inference."""
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "tmp/supplementary_metrics_v1"
OUT = ROOT / "results/supplementary_metrics_v1"
MODEL_ID = "FacebookAI/roberta-large"
MODEL_REVISION = "722cf37b1afa9454edce342e7895e588b6ff1d59"
MODEL_FILES = ("config.json", "model.safetensors", "vocab.json", "merges.txt", "tokenizer_config.json", "tokenizer.json")
WORDNET_REVISION = "96f9b3252457a2b97e52aec64c3dfceeb5c312d5"
WORDNET_URL = f"https://raw.githubusercontent.com/nltk/nltk_data/{WORDNET_REVISION}/packages/corpora/wordnet.zip"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def prepare():
    # Do not import the SFT loaders: their PEFT/Transformers 5 dependencies belong to a different process.
    import supplementary_metrics as metrics
    import bert_score
    import bert_score.scorer as scorer_module
    import bert_score.utils as utils
    import nltk
    from huggingface_hub import snapshot_download

    expected = {"bert-score": "0.3.13", "transformers": "4.46.3", "tokenizers": "0.20.3",
                "huggingface-hub": "0.36.2", "requests": "2.34.2", "charset-normalizer": "3.5.2", "urllib3": "2.8.0"}
    packages = (*expected, "nltk", "torch", "numpy", "pandas", "matplotlib", "safetensors", "certifi", "idna", "regex", "packaging", "filelock")
    versions = {name: importlib.metadata.version(name) for name in packages}
    assert all(versions[name] == version for name, version in expected.items()), "Use the isolated requirements-supplementary-metrics.txt stack"
    assert versions["nltk"] == "3.9.2"
    snapshot = LOCAL / "hub" / "models--FacebookAI--roberta-large" / "snapshots" / MODEL_REVISION
    if not all((snapshot / name).is_file() for name in MODEL_FILES):
        print(f"FETCH_METRIC_MODEL {MODEL_ID}@{MODEL_REVISION}", flush=True)
        fetched = snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(LOCAL / "hub"),
                                    allow_patterns=list(MODEL_FILES), token=False, max_workers=2)
        assert Path(fetched).resolve() == snapshot.resolve()
    assert all((snapshot / name).is_file() for name in MODEL_FILES)
    corpus = LOCAL / "nltk_data/corpora/wordnet.zip"
    if not corpus.is_file():
        print(f"FETCH_WORDNET {WORDNET_REVISION}", flush=True)
        corpus.parent.mkdir(parents=True, exist_ok=True)
        temporary = corpus.with_suffix(".tmp")
        urllib.request.urlretrieve(WORDNET_URL, temporary)
        with zipfile.ZipFile(temporary) as archive:
            assert archive.testzip() is None
            assert all(name.startswith("wordnet/") and ".." not in PurePosixPath(name).parts for name in archive.namelist())
        temporary.replace(corpus)
    nltk.data.path.insert(0, str(LOCAL / "nltk_data"))
    assert metrics.wordnet.get_version() == "3.0"
    # Unchanged Porter stems keep this probe focused on WordNet matching.
    assert metrics.meteor("car", ["auto"]) > 0
    tokenizer = utils.get_tokenizer(str(snapshot), use_fast=False)
    assert not tokenizer.is_fast and tokenizer.model_max_length == 512
    text = "hello " * 600
    full_length = len(tokenizer.encode(text.strip(), add_special_tokens=True, add_prefix_space=True, truncation=False))
    assert full_length > 512 and len(utils.sent_encode(tokenizer, text)) == 512
    assert utils.sent_encode(tokenizer, "") == tokenizer.build_inputs_with_special_tokens([])
    files = [snapshot / name for name in MODEL_FILES] + [corpus]
    assets = {"status": "prepared", "model": {"id": MODEL_ID, "revision": MODEL_REVISION,
              "snapshot": snapshot.relative_to(ROOT).as_posix()},
              "wordnet": {"revision": WORDNET_REVISION, "url": WORDNET_URL, "version": metrics.wordnet.get_version(),
                          "zip": corpus.relative_to(ROOT).as_posix()},
              "versions": versions, "bert_score_module_version": bert_score.__version__,
              "files": {path.relative_to(ROOT).as_posix(): {"sha256": sha(path), "bytes": path.stat().st_size} for path in files},
              "metric_source_sha256": {"prepare": sha(__file__), "helpers": sha(metrics.__file__),
                                       "bert_score_scorer": sha(scorer_module.__file__), "bert_score_utils": sha(utils.__file__)},
              "tokenizer": {"class": type(tokenizer).__name__, "is_fast": tokenizer.is_fast,
                            "model_max_length": tokenizer.model_max_length, "native_truncation_checked": True},
              "bertscore": {"num_layers": 17, "idf": False, "rescale_with_baseline": False, "dtype": "float32",
                            "reference_pooling": "independent maximum P/R/F1 over all complete references before native tokenizer truncation"},
              "meteor": {"alpha": .9, "beta": 3, "gamma": .5, "preprocess": "str.lower", "stemmer": "PorterStemmer", "reference_pooling": "maximum"},
              "word_tokenizer": "nltk.word_tokenize(preserve_line=True); original case for Distinct-4",
              "distinct4": "corpus unique/total 4grams, within response boundaries; denominator zero => null",
              "ece": "15 equal-width (lower, upper] bins on conditional binary-label confidence; no fitted temperature",
              "preparation_includes_gpu_inference": False}
    assets["fingerprint"] = hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest()
    destination = OUT / "assets.json"
    if destination.exists():
        assert read(destination) == assets, "Metric assets or environment changed; preserve existing results"
    else:
        write_json(destination, assets)
    print(json.dumps({"status": "prepared", "fingerprint": assets["fingerprint"], "model": assets["model"],
                      "wordnet_version": assets["wordnet"]["version"], "tokenizer": assets["tokenizer"]}), flush=True)


if __name__ == "__main__":
    LOCAL.mkdir(parents=True, exist_ok=True)
    with FileLock(str(LOCAL / "prepare.lock"), timeout=0):
        prepare()
