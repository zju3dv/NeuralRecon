"""CPU regression tests for the mesh score and scene report."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np
import open3d as o3d

from tools.evaluation_utils import eval_mesh
from tools.visualize_metrics import visualize


KEYS = {"dist1", "dist2", "prec", "recal", "fscore"}
ALIGNED = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
DISJOINT = [(0, 0, 1), (1, 0, 1), (0, 1, 1)]


def write_cloud(path, points):
    if not points:
        path.write_text("ply\nformat ascii 1.0\nelement vertex 0\n"
                        "property double x\nproperty double y\nproperty double z\n"
                        "end_header\n", encoding="ascii")
        return
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise OSError("Could not write " + str(path))


def reference(predicted, truth, threshold=.05):
    """Brute-force distances and directional match counts for nonempty clouds."""
    diff = np.asarray(predicted)[:, None, :] - np.asarray(truth)[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    pred_nearest = distances.min(axis=1)
    gold_nearest = distances.min(axis=0)
    precision = np.count_nonzero(pred_nearest < threshold) / len(predicted)
    recall = np.count_nonzero(gold_nearest < threshold) / len(truth)
    fscore = 2 / (1 / precision + 1 / recall) if precision and recall else 0.0
    return dict(dist1=pred_nearest.mean(), dist2=gold_nearest.mean(),
                prec=precision, recal=recall, fscore=fscore)


class MeshEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pred = self.root / "pred.ply"
        self.gold = self.root / "gold.ply"

    def evaluate(self, predicted, truth, **kwargs):
        write_cloud(self.pred, predicted)
        write_cloud(self.gold, truth)
        return eval_mesh(str(self.pred), str(self.gold), **kwargs)

    def assert_metrics(self, actual, expected):
        self.assertEqual(set(actual), KEYS)
        for key in sorted(KEYS):
            self.assertAlmostEqual(actual[key], expected[key], places=12, msg=key)

    def test_identical(self):
        self.assert_metrics(self.evaluate(ALIGNED, ALIGNED), reference(ALIGNED, ALIGNED))

    def test_partial_asymmetric(self):
        predicted = [ALIGNED[0], ALIGNED[1], (2, 3, 4)]
        truth = ALIGNED[:2]
        for first, second in ((predicted, truth), (truth, predicted)):
            with self.subTest(predicted_count=len(first)):
                self.assert_metrics(self.evaluate(first, second), reference(first, second))

    def test_disjoint_order_and_downsampling(self):
        expected = reference(ALIGNED, DISJOINT)
        for down_sample in (0, .02):
            for predicted, truth in ((ALIGNED, DISJOINT), (ALIGNED[::-1], DISJOINT[1:] + DISJOINT[:1])):
                with self.subTest(down_sample=down_sample, first_point=predicted[0]):
                    actual = self.evaluate(predicted, truth, down_sample=down_sample)
                    print("Disjoint points, voxel size {}: F-score {}".format(
                        down_sample, actual["fscore"]), flush=True)
                    self.assert_metrics(actual, expected)

    def test_threshold_excludes_exact_boundary(self):
        predicted = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        truth = [(0, .049, 0), (1, .05, 0), (2, .051, 0)]
        expected = reference(predicted, truth)
        self.assertEqual(expected["fscore"], 1 / 3)
        self.assert_metrics(self.evaluate(predicted, truth, down_sample=0), expected)

    def test_voxel_downsampling_merges_nearby_points(self):
        predicted = [(0, 0, 0), (.006, 0, 0), (.03, 0, 0)]
        truth = [(.003, 0, 0), (.03, 0, 0)]
        threshold = .001
        self.assert_metrics(
            self.evaluate(predicted, truth, threshold=threshold, down_sample=0),
            reference(predicted, truth, threshold),
        )
        self.assert_metrics(
            self.evaluate(predicted, truth, threshold=threshold, down_sample=.02),
            reference(truth, truth, threshold),
        )

    def test_empty_clouds_remain_undefined(self):
        for predicted, truth in (([], ALIGNED), (ALIGNED, []), ([], [])):
            for down_sample in (0, .02):
                with self.subTest(predicted_count=len(predicted), truth_count=len(truth),
                                  down_sample=down_sample), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    actual = self.evaluate(predicted, truth, down_sample=down_sample)
                    self.assertEqual(set(actual), KEYS)
                    self.assertTrue(all(np.isnan(value) for value in actual.values()))

    def test_nonfinite_clouds_are_not_valid_zero(self):
        for coordinate in (float("nan"), float("inf"), -float("inf")):
            invalid = [(coordinate, 0, 0), *ALIGNED[1:]]
            for predicted, truth in ((invalid, ALIGNED), (ALIGNED, invalid)):
                for down_sample in (0, .02):
                    with self.subTest(coordinate=coordinate, down_sample=down_sample,
                                      invalid_prediction=predicted is invalid):
                        try:
                            actual = self.evaluate(predicted, truth, down_sample=down_sample)
                        except (RuntimeError, IndexError):
                            # Open3D can reject invalid coordinates during sampling
                            # or return no neighbor for them.
                            continue
                        self.assertTrue(np.isnan(actual["fscore"]), actual)

    def test_triangle_mesh_ply_vertices_use_same_metric(self):
        for path, points in ((self.pred, ALIGNED), (self.gold, DISJOINT)):
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(points),
                o3d.utility.Vector3iVector([(0, 1, 2)]),
            )
            self.assertTrue(o3d.io.write_triangle_mesh(str(path), mesh))
        self.assert_metrics(eval_mesh(str(self.pred), str(self.gold)),
                            reference(ALIGNED, DISJOINT))

    def test_disjoint_scene_in_json_and_aggregate_report(self):
        aligned = self.evaluate(ALIGNED, ALIGNED)
        disjoint = self.evaluate(ALIGNED, DISJOINT)
        path = self.root / "metrics.json"
        path.write_text(json.dumps({"aligned": aligned, "disjoint": disjoint}),
                        encoding="utf-8")
        with redirect_stdout(io.StringIO()) as stdout:
            visualize(str(path))
        report = stdout.getvalue()
        rows = dict(line.split() for line in report.splitlines())
        print("Two-scene report: fscore {}".format(rows["fscore"]), flush=True)
        self.assertEqual(rows["prec"], "0.500")
        self.assertEqual(rows["recal"], "0.500")
        self.assertEqual(rows["fscore"], "0.500")
        scores = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(scores), {"aligned", "disjoint"})
        self.assertEqual(set(scores["disjoint"]), KEYS)
        self.assertIs(type(scores["disjoint"]["fscore"]), float)
        self.assertEqual(scores["disjoint"]["fscore"], 0.0)
        # JSON should contain a number, not the non-standard NaN literal.
        json.dumps(scores, allow_nan=False)
        cli = subprocess.check_output([
            sys.executable, "-B", "tools/visualize_metrics.py", "--model", str(self.root),
        ], text=True)
        self.assertEqual(cli, report)


if __name__ == "__main__":
    unittest.main()
