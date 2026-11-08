"""Synthetic polygon behavior checks for the preserved research functions."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
TEMP = REPO / ".testtmp"
TEMP.mkdir(exist_ok=True)

from shapely.geometry import box

from evaluate import main as evaluate
from research_core import apply_g, apply_g2, derive_g, exact


class ResearchCoreTests(unittest.TestCase):
    def test_cross_class_duplicate_suppression_restores_exact_match(self):
        polygon = box(0, 0, 1, 1)
        gt = [("MM", polygon, None)]
        predictions = [("HH", polygon, 0.78), ("MM", polygon, 0.97)]
        self.assertFalse(exact(gt, predictions))
        selected = apply_g2(predictions)
        self.assertEqual([row[0] for row in selected], ["MM"])
        self.assertTrue(exact(gt, selected))

    def test_high_confidence_wrong_class_can_remain(self):
        polygon = box(0, 0, 1, 1)
        gt = [("CC", polygon, None)]
        predictions = [("CC", polygon, 0.95), ("DD", polygon, 0.97)]
        selected = apply_g2(predictions)
        self.assertEqual([row[0] for row in selected], ["DD"])
        self.assertFalse(exact(gt, selected))

    def test_nms_suppresses_at_threshold_and_keeps_separate_boxes(self):
        overlapping = [("A", box(0, 0, 3, 1), 0.9), ("B", box(1, 0, 4, 1), 0.8)]
        self.assertEqual(len(apply_g2(overlapping)), 1)
        separate = [("A", box(0, 0, 1, 1), 0.9), ("B", box(2, 0, 3, 1), 0.8)]
        self.assertEqual(len(apply_g2(separate)), 2)

    def test_confidence_filter_leaves_unselected_classes_unchanged(self):
        polygon = box(0, 0, 1, 1)
        predictions = [("A", polygon, 0.5), ("A", polygon, 0.8), ("B", polygon, 0.4)]
        self.assertEqual([row[2] for row in apply_g(predictions, {"A": 0.7})], [0.8, 0.4])

    def test_validation_threshold_and_independent_evaluation(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            root = Path(folder)
            paths = {part: root / part for part in ("gt", "pred", "val_gt", "val_pred")}
            for path in paths.values():
                path.mkdir()
            square = "0 0 0 0.2 0 0.2 0.2 0 0.2"
            (paths["gt"] / "example.txt").write_text(square + "\n", encoding="utf-8")
            (paths["pred"] / "example.txt").write_text(square + " 0.9\n" + square.replace("0 ", "1 ", 1) + " 0.7\n", encoding="utf-8")
            (paths["val_gt"] / "example.txt").write_text(square + "\n", encoding="utf-8")
            (paths["val_pred"] / "example.txt").write_text(square + " 0.9\n0 0.7 0.7 0.9 0.7 0.9 0.9 0.7 0.9 0.5\n", encoding="utf-8")
            names = root / "classes.json"
            names.write_text(json.dumps(["MM", "HH"]), encoding="utf-8")
            thresholds = derive_g(str(paths["val_gt"]), str(paths["val_pred"]), {0: "MM", 1: "HH"})
            self.assertEqual(thresholds, {"MM": 0.7})
            output = root / "result.json"
            arguments = ["--gt", str(paths["gt"]), "--pred", str(paths["pred"]), "--val-gt", str(paths["val_gt"]), "--val-pred", str(paths["val_pred"]), "--classes", str(names), "--post", "all", "--output", str(output)]
            self.assertEqual(evaluate(arguments), 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(result["independent_postprocessing"])
            self.assertEqual(result["results"]["none"]["exact_images"], 0)
            self.assertEqual(result["results"]["confidence"]["exact_images"], 0)
            self.assertEqual(result["results"]["nms"]["exact_images"], 1)
            before = output.read_bytes()
            with self.assertRaises(ValueError):
                evaluate(arguments)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
