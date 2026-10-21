import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from budget_router.core import digest, read, write
from budget_router.experiment import configuration, run, verify


class FakeReranker:
    """Test double only; never used for published measurements."""

    def __init__(self, config, cache):
        pass

    def score(self, question, docs):
        return [
            float(len(set(question.lower().split()) & set(d["text"].lower().split()))) for d in docs
        ]


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)
        cls.root = Path(".").resolve()
        with (
            patch("budget_router.experiment.LocalReranker", FakeReranker),
            patch("budget_router.experiment.importlib.metadata.version", return_value="test"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run(cls.root, cls.output / "dev", "dev", cls.output / "cache", None)
            run(
                cls.root,
                cls.output / "eval",
                "eval",
                cls.output / "cache",
                cls.output / "dev/policies.json",
            )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_round_trip(self):
        self.assertEqual(verify(self.root, self.output / "dev")["queries"], 12)
        self.assertEqual(
            set(read(self.output / "dev/metadata.json")["input_hashes"]),
            {"data/corpus.json", "data/queries.json", "config/experiment.json"},
        )
        self.assertEqual(verify(self.root, self.output / "eval")["samples"], 624)

    def test_refuse_existing_output_and_missing_policy(self):
        with self.assertRaises(FileExistsError):
            run(self.root, self.output / "dev", "dev", self.output, None)
        with self.assertRaises(ValueError):
            run(self.root, self.output / "new", "eval", self.output, None)
        self.assertFalse((self.output / "new").exists())

    def test_tampering_with_rehashed_evidence(self):
        output = self.output / "eval"
        cases = [
            ("summary.json", lambda x: x["frontier"].append("fake")),
            ("counterfactual.json", lambda x: x[0]["signals"].update(confidence=0.999)),
            ("policies.json", lambda x: x["policies"][0].update(low=1)),
            ("errors.json", lambda x: x.append({"fake": 1})),
        ]
        for filename, mutate in cases:
            path = output / filename
            original = path.read_bytes()
            manifest = (output / "manifest.json").read_bytes()
            try:
                value = read(path)
                mutate(value)
                write(path, value)
                hashes = read(output / "manifest.json")
                hashes[filename] = digest(path)
                write(output / "manifest.json", hashes)
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    verify(self.root, output)
            finally:
                path.write_bytes(original)
                (output / "manifest.json").write_bytes(manifest)

    def test_trace_budget_and_duplicate_validation(self):
        output = self.output / "eval"
        path = output / "trace.jsonl"
        original = path.read_bytes()
        manifest = (output / "manifest.json").read_bytes()
        for mutation in ("depth", "duplicate", "nan", "missing"):
            try:
                rows = [json.loads(line) for line in original.decode().splitlines()]
                if mutation == "depth":
                    rows[0]["depth"] = 99
                elif mutation == "duplicate":
                    rows.append(rows[0])
                elif mutation == "nan":
                    rows[0]["elapsed_ms"] = float("nan")
                else:
                    rows.pop()
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                hashes = read(output / "manifest.json")
                hashes["trace.jsonl"] = digest(path)
                write(output / "manifest.json", hashes)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    verify(self.root, output)
            finally:
                path.write_bytes(original)
                (output / "manifest.json").write_bytes(manifest)

    def test_configuration_rejects_unpinned_revision(self):
        path = self.output / "bad-config.json"
        c = read(self.root / "config/experiment.json")
        c["revision"] = "main"
        write(path, c)
        with self.assertRaises(ValueError):
            configuration(path)

    def test_cli_routes_and_reports_errors(self):
        command = [
            sys.executable,
            "-m",
            "budget_router",
            "route",
            "How to restore a notebook?",
            "--policies",
            str(self.output / "dev/policies.json"),
        ]
        result = subprocess.run(command + ["--budget", "0"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["depth"], 0)
        result = subprocess.run(command + ["--budget", "-1"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("nonnegative", result.stderr)

    def test_cli_verify_and_unknown_policy(self):
        from budget_router.__main__ import main

        with (
            patch("sys.argv", ["budget-router", "verify", "--output", str(self.output / "eval")]),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(main(), 0)
            self.assertTrue(json.loads(output.getvalue())["verified"])
        with (
            patch(
                "sys.argv",
                [
                    "budget-router",
                    "route",
                    "notebook",
                    "--policy",
                    "missing",
                    "--policies",
                    str(self.output / "dev/policies.json"),
                ],
            ),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as exc,
        ):
            main()
        self.assertEqual(exc.exception.code, 2)

    def test_replay_in_fresh_process(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "budget_router",
                "verify",
                "--output",
                str(self.output / "eval"),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
