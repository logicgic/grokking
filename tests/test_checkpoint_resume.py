"""End-to-end regression test for exact CPU checkpoint/resume training.

Run from the repository root with:
    .venv312/Scripts/python.exe -m unittest discover -s tests -p test_checkpoint_resume.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Mapping

import torch


REPOSITORY = Path(__file__).resolve().parents[1]


class CheckpointResumeTest(unittest.TestCase):
    """Exercise the real Hydra CLI in three independent training processes."""

    def assert_state_equal(self, expected: object, actual: object, path: str) -> None:
        """Compare nested state exactly, including tensors without tolerances."""
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor, path)
            self.assertEqual(expected.dtype, actual.dtype, path)
            self.assertEqual(expected.shape, actual.shape, path)
            self.assertTrue(torch.equal(expected, actual), f"{path}: tensors differ")
        elif isinstance(expected, Mapping):
            self.assertIsInstance(actual, Mapping, path)
            self.assertEqual(set(expected), set(actual), path)
            for key, value in expected.items():
                self.assert_state_equal(value, actual[key], f"{path}.{key}")
        elif isinstance(expected, (list, tuple)):
            self.assertIsInstance(actual, type(expected), path)
            self.assertEqual(len(expected), len(actual), path)
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.assert_state_equal(left, right, f"{path}[{index}]")
        else:
            self.assertEqual(expected, actual, path)

    def run_training(
        self,
        directory: Path,
        max_steps: int,
        *,
        resume_from: Path | None = None,
        seed: int = 17,
        overrides: tuple[str, ...] = (),
        expected_error: str | None = None,
    ) -> tuple[dict, dict, str]:
        directory.mkdir(parents=True)
        checkpoint_dir = directory / "checkpoints"
        command = [
            sys.executable,
            str(REPOSITORY / "scripts" / "train_grokk.py"),
            "wandb.use_wandb=false",
            "dataset.p=11",
            "dataset.frac_train=0.4",
            "model.transformer_config.heads=2",
            "model.transformer_config.hidden_dim=16",
            "model.transformer_config.attn_dim=8",
            "model.transformer_config.intermediate_dim=32",
            "model.transformer_config.num_blocks=1",
            # Nonzero dropout makes torch RNG restoration observable.
            "model.transformer_config.dropout=0.2",
            f"train.seed={seed}",
            "train.bsize=8",
            "train.num_workers=0",
            "train.eval_every=2",
            "train.eval_batches=1",
            "train.warmup_steps=10",
            "train.save_every=3",
            f"train.max_steps={max_steps}",
            "train.checkpoint_dir=checkpoints",
            "hydra.run.dir=.",
            "hydra.output_subdir=null",
            "hydra.job.chdir=false",
        ]
        if resume_from is not None:
            # Hydra receives one argv element even when a path contains spaces.
            command.append(f"train.resume_from='{resume_from.as_posix()}'")
        command.extend(overrides)
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES="",
            WANDB_MODE="disabled",
            WANDB_DISABLED="true",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            PYTHONHASHSEED="0",
            PYTHONIOENCODING="utf-8",
            HYDRA_FULL_ERROR="1",
        )
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(REPOSITORY / "grokk_replica"), str(REPOSITORY)]
            + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
        )
        try:
            result = subprocess.run(
                command,
                cwd=directory,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            self.fail(f"Training timed out after 120 seconds: {error.stdout!r}")
        if expected_error is not None:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn(expected_error, result.stdout)
            self.assertFalse(
                checkpoint_dir.exists() and any(checkpoint_dir.iterdir()),
                "Rejected resume must not create checkpoint artifacts",
            )
            return {}, {}, result.stdout
        self.assertEqual(result.returncode, 0, result.stdout)
        latest = checkpoint_dir / "latest.pt"
        final = checkpoint_dir / "model-final.pt"
        for artifact in (latest, final):
            self.assertTrue(artifact.is_file(), f"Missing {artifact}\n{result.stdout}")
            self.assertGreater(artifact.stat().st_size, 0, str(artifact))
        self.assertFalse(list(checkpoint_dir.glob("*.tmp")), result.stdout)
        # These complete checkpoints were created locally by this test.
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        weights = torch.load(final, map_location="cpu", weights_only=True)
        self.assertTrue(
            {"model", "optimizer", "scheduler", "step", "config", "dataset", "rng"}
            <= checkpoint.keys(),
            f"Incomplete checkpoint: {checkpoint.keys()}",
        )
        self.assertEqual(checkpoint["step"], max_steps, result.stdout)
        self.assertEqual(checkpoint["config"]["train"]["max_steps"], max_steps)
        self.assertFalse(checkpoint["config"]["wandb"]["use_wandb"])
        self.assertTrue(weights, "Final model state must contain parameters")
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in weights.values()))
        self.assertTrue(checkpoint["optimizer"]["state"], "Optimizer state is empty")
        self.assertEqual(checkpoint["scheduler"]["last_epoch"], max_steps)
        self.assertIsInstance(checkpoint["dataset"], dict)
        self.assertTrue({"python", "torch"} <= checkpoint["rng"].keys())
        self.assert_state_equal(checkpoint["model"], weights, "saved model vs weights")
        return checkpoint, weights, result.stdout

    def test_four_steps_match_two_steps_then_resume(self) -> None:
        # Spaces also exercise subprocess/Hydra path handling on Windows.
        with tempfile.TemporaryDirectory(prefix="grokk checkpoint resume ") as temporary:
            root = Path(temporary)
            continuous, continuous_weights, continuous_output = self.run_training(
                root / "continuous", 4
            )
            partial, partial_weights, _ = self.run_training(root / "partial", 2)
            resume_path = root / "partial" / "checkpoints" / "latest.pt"
            resumed, resumed_weights, resumed_output = self.run_training(
                root / "resumed", 4, resume_from=resume_path, seed=991
            )

            self.assertIn("Saved checkpoint at step 3:", continuous_output)
            self.assertIn("Resumed checkpoint at step 2:", resumed_output)
            self.assertIn("Saved checkpoint at step 3:", resumed_output)
            self.assertEqual(resumed["config"]["train"]["seed"], 991)
            self.assertEqual(
                Path(resumed["config"]["train"]["resume_from"]), resume_path
            )
            self.assert_state_equal(
                continuous_weights, resumed_weights, "continuous vs resumed weights"
            )
            for key in ("optimizer", "scheduler", "rng", "dataset"):
                with self.subTest(state=key):
                    self.assert_state_equal(continuous[key], resumed[key], key)
            self.assert_state_equal(partial["dataset"], resumed["dataset"], "restored dataset")
            self.assertTrue(
                any(
                    not torch.equal(partial_weights[key], resumed_weights[key])
                    for key in partial_weights
                ),
                "Resume must perform training beyond step 2",
            )
            with self.subTest(rejection="dataset configuration mismatch"):
                self.run_training(
                    root / "rejected dataset",
                    4,
                    resume_from=resume_path,
                    overrides=("dataset.p=13",),
                    expected_error="Resume dataset configuration differs from checkpoint",
                )
            with self.subTest(rejection="budget below checkpoint step"):
                self.run_training(
                    root / "rejected budget",
                    1,
                    resume_from=resume_path,
                    expected_error="max_steps must be at least the checkpoint step",
                )
            # Rejected runs must also leave the source checkpoint intact.
            self.assert_state_equal(
                partial,
                torch.load(resume_path, map_location="cpu", weights_only=False),
                "source checkpoint after rejected resumes",
            )
            train_pairs = resumed["dataset"]["train_pairs"]
            val_pairs = resumed["dataset"]["val_pairs"]
            self.assertEqual(len(train_pairs), int(11 * 11 * 0.4))
            self.assertEqual(len(val_pairs), 11 * 11 - len(train_pairs))
            self.assertFalse(set(train_pairs) & set(val_pairs))
            self.assertEqual(set(train_pairs) | set(val_pairs), set(range(11 * 11)))


if __name__ == "__main__":
    unittest.main()
