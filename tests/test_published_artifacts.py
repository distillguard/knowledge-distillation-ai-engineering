"""Keep published CSV bytes consistent with their declared SHA on every checkout."""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PublishedArtifactTests(unittest.TestCase):
    def test_committed_per_item_csv_bytes_match_declared_hashes(self):
        tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0"))
        checked = 0
        for relative in sorted(tracked):
            if not relative.startswith("results/") or not relative.endswith(".json"):
                continue
            record = json.loads((ROOT / relative).read_text(encoding="utf-8"))
            if not isinstance(record, dict) or "per_item_csv_sha256" not in record:
                continue
            csv_path, expected = record["per_item_csv"], record["per_item_csv_sha256"]
            self.assertIn(csv_path, tracked, relative)
            indexed = subprocess.check_output(["git", "cat-file", "blob", ":" + csv_path], cwd=ROOT)
            self.assertEqual(hashlib.sha256(indexed).hexdigest(), expected, csv_path + " Git bytes")
            self.assertEqual(hashlib.sha256((ROOT / csv_path).read_bytes()).hexdigest(), expected, csv_path + " checkout bytes")
            checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
