"""Portable CLI checks using only fixtures generated during each test run."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from fmri_response.cli import plain, response_demo

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
        cls.workdir = temporary / "different working directory"
        cls.workdir.mkdir()
        cls.private_data = cls.workdir / "generated fixture data"
        cls.private_data.mkdir()
        # Entirely synthetic inputs and coefficients; no study files are read.
        fields = np.random.default_rng(310).normal(size=(6, 3, 5, 4))
        np.savez(cls.private_data / "msc_sessions123.npz", responses=fields)
        np.save(cls.private_data / "locked_adni_mean_history.npy", np.zeros((4, 3)))
        np.save(cls.private_data / "locked_adni_mean_target.npy", np.zeros(4))
        parameters = {"personal4": [0.0] * 4, "base3": [0.0] * 3}
        (cls.private_data / "fixed_coefficients.json").write_text(json.dumps(parameters))
        mean_loss = float(np.mean((fields[:, 2, 4] - fields[:, 2, 0]) ** 2))
        contrast = {
            "candidate_mean_loss": mean_loss,
            "reference_mean_loss": mean_loss,
            "mean_reduction": 0.0,
            "relative_mean_reduction": 0.0,
            "participants_improved": 0,
            "participant_differences": [0.0] * 6,
        }
        cls.expected_transfer = {"n": 6, **{
            key: contrast for key in (
                "personal_vs_base3", "personal_vs_fixed_adni_mean_history",
                "personal_vs_adni_mean_target",
            )
        }}
        (cls.private_data / "msc_expected.json").write_text(json.dumps(cls.expected_transfer))
        (cls.private_data / "adni_aggregate.json").write_text(
            json.dumps({"scope": "Synthetic test fixture"})
        )
        # This unrelated file must never be swept into summary output.
        (cls.private_data / "unrelated.json").write_text('{"not_a_summary": true}')
        with threadpool_limits(limits=1):
            cls.expected_demo = plain(response_demo())

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

    def test_demo_runs_without_private_data_after_relocation(self):
        actual = json.loads(self.invoke("--mode", "demo").stdout)
        self.assert_result_equal(actual, self.expected_demo)

    def test_transfer_uses_explicit_data_directory_with_synthetic_arrays(self):
        actual = json.loads(self.invoke(
            "--mode", "msc", "--data-dir", self.private_data.name
        ).stdout)
        replay_error = actual.pop("maximum_mean_loss_replay_error")
        self.assert_result_equal(actual, self.expected_transfer)
        self.assertLessEqual(replay_error, 1e-12)

    def test_summary_and_nested_output_path_after_relocation(self):
        output = Path("nested results/summary.json")
        receipt = json.loads(self.invoke(
            "--mode", "summary", "--data-dir", self.private_data.name,
            "--output", str(output),
        ).stdout)
        self.assertEqual(receipt["output"], str(output))
        actual = json.loads((self.workdir / output).read_text())
        expected = {
            name: json.loads((self.private_data / name).read_text())
            for name in ("adni_aggregate.json", "fixed_coefficients.json", "msc_expected.json")
        }
        self.assert_result_equal(actual, expected)
        self.assertFalse((self.project / output).exists())

    def test_missing_private_data_reports_access_instructions(self):
        for mode in ("msc", "summary"):
            with self.subTest(mode=mode):
                result = self.invoke("--mode", mode, success=False)
                self.assertIn("Private data are unavailable", result.stderr)
                self.assertIn("git submodule update --init data", result.stderr)
                self.assertIn("--data-dir", result.stderr)
                if mode == "msc":
                    self.assertIn("gh release download", result.stderr)
                    self.assertIn("--repo yury-rd/fmri-transfer-data", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_empty_explicit_data_directory_is_not_an_empty_successful_summary(self):
        result = self.invoke("--mode", "summary", "--data-dir", str(self.workdir), success=False)
        self.assertIn("adni_aggregate.json", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_local_input_is_relative_to_working_directory(self):
        rng = np.random.default_rng(20260918)
        np.savez(
            self.workdir / "local fields.npz",
            responses=rng.normal(size=(7, 3, 5, 13)),
            reliability=rng.normal(size=(7, 2)),
            design=rng.normal(size=(7, 3)),
        )
        actual = json.loads(self.invoke("--input", "local fields.npz").stdout)
        expected = {**self.expected_demo, "scope": "Local user-supplied response fields"}
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
        self.assertIn("--data-dir", help_text)
        for other_paper_option in ("--case", "--branch", "--splits"):
            self.assertNotIn(other_paper_option, help_text)


if __name__ == "__main__":
    unittest.main()
