import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="fmri-response-tests-")
        cls.addClassCleanup(cls.temporary.cleanup)
        temporary = Path(cls.temporary.name)
        cls.project = temporary / "renamed project with spaces"
        cls.project.mkdir()
        shutil.copy2(ROOT / "run.py", cls.project / "run.py")
        shutil.copytree(
            ROOT / "fmri_response", cls.project / "fmri_response",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copytree(ROOT / "data", cls.project / "data")
        cls.workdir = temporary / "different working directory"
        cls.workdir.mkdir()

    def invoke(self, *arguments, success=True):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, str(self.project / "run.py"), *arguments],
            cwd=self.workdir, env=environment, capture_output=True, text=True,
            timeout=120,
        )
        if success:
            self.assertEqual(completed.returncode, 0, completed.stderr)
        else:
            self.assertNotEqual(completed.returncode, 0)
        return completed

    def assert_result_equal(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_result_equal(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for value, reference in zip(actual, expected):
                self.assert_result_equal(value, reference)
        elif isinstance(expected, (int, float)):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        else:
            self.assertEqual(actual, expected)

    def test_demo_reproduces_reference_after_relocation(self):
        expected = json.loads((ROOT / "tests/expected_demo.json").read_text())
        actual = json.loads(self.invoke("--mode", "demo").stdout)
        self.assert_result_equal(actual, expected)

    def test_msc_reproduces_released_results_after_relocation(self):
        expected = json.loads((ROOT / "data/msc_expected.json").read_text())
        actual = json.loads(self.invoke("--mode", "msc").stdout)
        self.assertEqual(actual["n"], expected["n"])
        for key in expected:
            if key.startswith("personal_"):
                self.assert_result_equal(actual[key], expected[key])
        self.assertLessEqual(actual["maximum_mean_loss_replay_error"], 1e-12)

    def test_summary_and_nested_output_path_after_relocation(self):
        output = Path("nested results/summary.json")
        receipt = json.loads(self.invoke("--mode", "summary", "--output", str(output)).stdout)
        self.assertEqual(receipt["output"], str(output))
        actual = json.loads((self.workdir / output).read_text())
        expected = {
            path.name: json.loads(path.read_text())
            for path in sorted((ROOT / "data").glob("*.json"))
        }
        self.assert_result_equal(actual, expected)
        self.assertFalse((self.project / output).exists())

    def test_local_input_is_relative_to_working_directory(self):
        rng = np.random.default_rng(20260918)
        np.savez(
            self.workdir / "local fields.npz",
            responses=rng.normal(size=(7, 3, 5, 13)),
            reliability=rng.normal(size=(7, 2)),
            design=rng.normal(size=(7, 3)),
        )
        actual = json.loads(self.invoke("--input", "local fields.npz").stdout)
        expected = json.loads((ROOT / "tests/expected_demo.json").read_text())
        expected["scope"] = "Local user-supplied response fields"
        self.assert_result_equal(actual, expected)

    def test_missing_input_reports_the_requested_path(self):
        result = self.invoke("--input", "missing fields.npz", success=False)
        self.assertIn("missing fields.npz", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_invalid_local_axes_are_rejected(self):
        np.savez(
            self.workdir / "invalid.npz", responses=np.zeros((7, 5, 3, 13)),
            reliability=np.zeros((7, 2)), design=np.zeros((7, 3)),
        )
        result = self.invoke("--input", "invalid.npz", success=False)
        self.assertIn("shape (participants, 3, 5, edges)", result.stderr)

    def test_help_contains_only_this_papers_options(self):
        help_text = self.invoke("--help").stdout
        self.assertIn("demo,msc,summary", help_text)
        for other_paper_option in ("--case", "--branch", "--splits"):
            self.assertNotIn(other_paper_option, help_text)


if __name__ == "__main__":
    unittest.main()
