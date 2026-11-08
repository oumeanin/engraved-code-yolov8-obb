"""Small synthetic checks; these do not reproduce the paper's experiments."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "code"
TEMP = REPO / ".testtmp"
TEMP.mkdir(exist_ok=True)
sys.path.insert(0, str(CODE))

from cli_support import fresh_path
from oversample import replicate
from restore_split import restore
from train import main as train


class TemporaryInputs(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=TEMP)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def dataset(self):
        source = self.root / "source"
        for part in ("train", "val", "test"):
            (source / "images" / part).mkdir(parents=True)
            (source / "labels" / part).mkdir(parents=True)
            (source / "images" / part / f"{part}.bmp").write_bytes(b"synthetic image bytes")
            (source / "labels" / part / f"{part}.txt").write_text("0 0 0 0.2 0 0.2 0.2 0 0.2\n", encoding="utf-8")
        return source


class CommandTests(TemporaryInputs):
    def test_train_refuses_output_inside_external_split_directory(self):
        import yaml

        source = self.dataset()
        logical_root = self.root / "logical-root"
        logical_root.mkdir()
        data = self.root / "dataset.yaml"
        data.write_text(yaml.safe_dump({
            "path": str(logical_root),
            **{part: str(source / "images" / part) for part in ("train", "val", "test")},
        }), encoding="utf-8")
        weights = self.root / "fake.pt"
        weights.write_bytes(b"synthetic placeholder")
        output = source / "images" / "train" / "new-run"
        fake_model = SimpleNamespace(train=lambda **kwargs: None)
        with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=lambda _: fake_model)}):
            with self.assertRaises(ValueError):
                train(["--experiment", "a", "--data", str(data), "--weights", str(weights), "--output", str(output)])
        self.assertFalse(output.exists())

    def test_all_help_commands_work_without_site_packages(self):
        for name in ("train", "predict", "evaluate", "oversample", "restore_split"):
            with self.subTest(command=name):
                result = subprocess.run(
                    [sys.executable, "-B", "-S", str(CODE / f"{name}.py"), "--help"],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_input_and_output_overlap_is_rejected(self):
        source = self.dataset()
        with self.assertRaises(ValueError):
            fresh_path(source / "nested", (source,))
        with self.assertRaises(ValueError):
            fresh_path(source, (source,))

    def test_largest_multiplier_repeats_whole_image_and_labels(self):
        source = self.dataset()
        label = source / "labels" / "train" / "train.txt"
        label.write_text("0 0 0 0.2 0 0.2 0.2 0 0.2\n1 0.3 0 0.5 0 0.5 0.2 0.3 0.2\n", encoding="utf-8")
        before = label.read_bytes()
        output = self.root / "oversampled"
        summary = replicate(source, output, {0: "ZZ", 1: "DD"})
        self.assertEqual(summary["output_train_images"], 10)
        self.assertEqual(len(list((output / "images" / "train").glob("*.bmp"))), 10)
        copied_labels = list((output / "labels" / "train").glob("*.txt"))
        self.assertEqual(len(copied_labels), 10)
        self.assertTrue(all(path.read_bytes() == before for path in copied_labels))
        self.assertEqual(label.read_bytes(), before)
        self.assertEqual(len(list((source / "images" / "train").glob("*.bmp"))), 1)
        for part in ("val", "test"):
            self.assertEqual((output / "labels" / part / f"{part}.txt").read_bytes(),
                             (source / "labels" / part / f"{part}.txt").read_bytes())
        with self.assertRaises(ValueError):
            replicate(source, output, {0: "ZZ", 1: "DD"})

    def test_restore_uses_manifest_and_refuses_missing_input(self):
        source = self.dataset()
        parts = {part: {"count": 1, "files": [part]} for part in ("train", "val", "test")}
        hashes = {part: hashlib.sha256(part.encode()).hexdigest()[:16] for part in parts}
        manifest = self.root / "split.json"
        manifest.write_text(json.dumps({"splits": parts, "split_hash": hashes}), encoding="utf-8")
        output = self.root / "restored"
        self.assertEqual(restore(manifest, source, output), {"train": 1, "val": 1, "test": 1})
        self.assertTrue((output / "labels" / "train" / "train.txt").is_file())
        with self.assertRaises(ValueError):
            restore(manifest, source, output)
        (source / "labels" / "test" / "test.txt").unlink()
        missing_output = self.root / "missing"
        with self.assertRaises(ValueError):
            restore(manifest, source, missing_output)
        self.assertFalse(missing_output.exists())


if __name__ == "__main__":
    unittest.main()
