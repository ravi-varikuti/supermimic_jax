import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wandb_logging import WandbLogger


class WandbLoggingTests(unittest.TestCase):
    def test_disabled_logging_does_not_require_sdk(self):
        with patch.dict("sys.modules", {"wandb": None}):
            logger = WandbLogger("/tmp", {"config": {}})
            logger.log({"iteration": 1})
            logger.finish()
            self.assertIsNone(logger.run)
            logger = WandbLogger("/tmp", {"config": {}, "wandb": {"id": "saved"}}, mode="disabled")
            self.assertIsNone(logger.run)

    @unittest.skipUnless(importlib.util.find_spec("wandb"), "Optional W&B SDK is not installed")
    def test_real_offline_metrics_evaluation_and_saved_identity(self):
        metadata = {"config": {"seed": 0, "num_envs": 4}}
        evaluation = {"num_motions": 1, "failures": 0, "mean_normalized_return": .9,
                      "clips": [{"motion": "walk", "normalized_return": .9, "failed": False}]}
        with tempfile.TemporaryDirectory() as directory:
            logger = WandbLogger(directory, metadata, enabled=True, mode="offline",
                                 project="supermimic-test", name="offline-test")
            try:
                logger.log({"iteration": 5, "simulator_steps": 100, "actor_loss": .1,
                            "world_model_loss": None}, evaluation)
                self.assertEqual(logger.run.summary["eval/failures"], 0)
                self.assertEqual(logger.run.config["num_envs"], 4)
                identity = dict(logger.identity)
            finally:
                logger.finish()
            self.assertTrue(list(Path(directory).glob("wandb/offline-run-*/*.wandb")))
            metadata["wandb"] = identity
            with self.assertRaises(ValueError):
                WandbLogger(directory, metadata, mode="offline", project="different-project")
            # SDK offline sessions are separate segments, but retain logical run ID.
            resumed = WandbLogger(directory, metadata, mode="offline")
            try:
                self.assertEqual(resumed.identity["id"], identity["id"])
                resumed.log({"iteration": 6, "simulator_steps": 120, "actor_loss": .09})
            finally:
                resumed.finish()


if __name__ == "__main__":
    unittest.main()
