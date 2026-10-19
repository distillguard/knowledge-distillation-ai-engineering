"""Fetch only pinned GPT-2/Dolly sources and inputs for the current study."""
import hashlib
import json
from pathlib import Path
import urllib.request

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/soft_gate_gradient_v1"
HUB = ROOT / "tmp/hf_home/hub"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fetch(repo, revision, files, repo_type="model"):
    prefix = "models--" if repo_type == "model" else "datasets--"
    folder = HUB / (prefix + repo.replace("/", "--")) / "snapshots" / revision
    if not all((folder / name).is_file() for name in files):
        print("FETCH_PINNED", repo, revision, flush=True)
        snapshot_download(repo, revision=revision, repo_type=repo_type, cache_dir=HUB,
                          allow_patterns=files, token=False, max_workers=2)
    assert all((folder / name).is_file() for name in files)
    return folder


def main():
    sources = read(OUT / "official_sources.json")
    commit = sources["commit"]
    for name, item in sources["files"].items():
        url = f"https://raw.githubusercontent.com/microsoft/LMOps/{commit}/minillm/{name}"
        assert item["url"] == url and ".." not in Path(name).parts
        path = ROOT / ".sources" / ("minillm-" + commit) / "minillm" / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read()
            assert hashlib.sha256(content).hexdigest() == item["sha256"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    spec = read(OUT / "preparation.json")
    model_files = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"]
    for model in spec["models"].values():
        fetch(model["id"], model["revision"], model_files)
    teacher = read(OUT / "teacher.json")
    folder = fetch(teacher["id"], teacher["revision"], list(teacher["files"]))
    for name, item in teacher["files"].items():
        with (folder / name).open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == item["sha256"]
    data = spec["dataset"]
    folder = fetch(data["id"], data["revision"], ["raw.jsonl", "valid.jsonl"], "dataset")
    for name, key in (("raw.jsonl", "raw_sha256"), ("valid.jsonl", "evaluation_sha256")):
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == data[key]
    audit = read(ROOT / "results/dolly_sft_v1/evaluation_preflight.json")["classification"]
    for dataset, filename in (("sst2", "data/validation-00000-of-00001.parquet"),
                              ("boolq", "boolq/validation-00000-of-00001.parquet")):
        item = audit[dataset]
        folder = fetch(item["repo"], item["revision"], [filename], "dataset")
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == item["sha256"]
    print("CURRENT_STUDY_ASSETS_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
