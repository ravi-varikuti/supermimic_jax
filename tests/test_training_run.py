from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from typing import NamedTuple

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from checkpoints import checkpoint_metadata, load_checkpoint, save_checkpoint
from evaluation import Evaluator, clip_metrics
from reset_sampling import ReturnSampler
from schedules import learning_rate_schedule
from train_config import RunConfig, preset
from transitions import Transition


def records(rewards, steps, frames, done):
    rewards = jnp.asarray(rewards, jnp.float32)[:, None]
    shape = rewards.shape
    fields = jnp.zeros(shape + (1,))
    return Transition(fields, fields, fields, rewards, jnp.asarray(done)[:, None],
                      jnp.zeros(shape, bool), fields, fields, fields, fields,
                      {"episode_steps": jnp.asarray(steps)[:, None],
                       "reference_frame": jnp.asarray(frames)[:, None],
                       "motion_id": jnp.zeros(shape, jnp.int32)})


class TrainingRunTests(unittest.TestCase):
    def test_cosine_floor_and_optimizer_substep_counting(self):
        schedule = learning_rate_schedule(2e-4, 100, updates_per_iteration=10)
        self.assertAlmostEqual(float(schedule(0)), 2e-4, places=10)
        self.assertAlmostEqual(float(schedule(500)), 2e-4 * .515, places=10)
        self.assertAlmostEqual(float(schedule(1000)), 2e-4 * .03, places=10)
        self.assertAlmostEqual(float(schedule(2000)), 2e-4 * .03, places=10)

    def test_checkpoint_exact_optimizer_rng_and_host_statistics_resume(self):
        optimizer = optax.adam(learning_rate_schedule(.01, 10))
        train = TrainState.create(apply_fn=lambda p, x: p * x, params={"x": jnp.ones(2)}, tx=optimizer)
        initial_train = train
        train = train.apply_gradients(grads={"x": jnp.array([.2, .4])})
        template = {"train": train, "rng": jax.random.key(9),
                    "host": np.array([.123456789012345], np.float64),
                    "simulator_steps": np.array(4_000_000_000, np.int64)}
        metadata = {"config": {"batch": 2}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.ckpt"
            save_checkpoint(path, template, metadata)
            self.assertEqual(checkpoint_metadata(path), metadata)
            restored = load_checkpoint(path, template | {"train": initial_train}, metadata)
            np.testing.assert_array_equal(restored["host"], template["host"])
            self.assertEqual(restored["host"].dtype, np.float64)
            self.assertEqual(int(restored["simulator_steps"]), 4_000_000_000)
            np.testing.assert_array_equal(jax.random.key_data(restored["rng"]), jax.random.key_data(template["rng"]))
            gradient = {"x": jnp.array([.3, -.4])}
            expected = train.apply_gradients(grads=gradient)
            actual = restored["train"].apply_gradients(grads=gradient)
            for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
                np.testing.assert_array_equal(a, b)
            with self.assertRaises(ValueError):
                load_checkpoint(path, template, {"different": True})
            with self.assertRaises(ValueError):
                load_checkpoint(path, {"x": jnp.ones(2)})

    def test_reset_mixture_bins_and_completed_returns_across_chunks(self):
        sampler = ReturnSampler([5, 3], 2., 1, 5, discount=.9, max_reward=1.)
        sample = jax.jit(lambda key, weights: sampler.sample_jax(key, weights, 1000))
        ids, frames = sample(jax.random.key(0), sampler.probabilities())
        self.assertTrue(bool(((frames >= 0) & (frames < jnp.array([4, 2])[ids])).all()))
        sampler.observe(records([1., 1.], [0, 1], [0, 1], [False, False]))
        np.testing.assert_array_equal(sampler.state["visits"], 0)
        sampler.observe(records([1.], [2], [2], [True]))
        np.testing.assert_allclose(sampler.state["estimate"], [(2.71 + 1.9) / 2, 1., 0.])
        self.assertEqual(int(sampler.state["pending_length"][0]), 0)
        weights = np.asarray(sampler.probabilities())
        self.assertAlmostEqual(float(weights.sum()), 1., places=6)
        self.assertGreater(weights[2], weights[0])
        self.assertTrue((weights >= .5 / 3).all())
        sampler.observe(records([np.nan], [0], [0], [True]))
        self.assertTrue(np.isfinite(sampler.probabilities()).all())

    def test_evaluation_denominator_does_not_shrink_after_failure(self):
        result = clip_metrics(1., 10, True, expected_steps=100, max_reward=.1)
        self.assertAlmostEqual(result["normalized_return"], .1)
        self.assertTrue(result["failed"])
        result = clip_metrics(10., 100, False, expected_steps=100, max_reward=.1)
        self.assertAlmostEqual(result["normalized_return"], 1.)

    def test_presets_and_json_config_roundtrip(self):
        for name in ("smoke", "local", "paper"):
            config = preset(name)
            self.assertEqual(RunConfig(**config.to_dict()), config)
        self.assertEqual(preset("paper").num_envs, 4096)
        self.assertEqual(preset("paper").dynamics_hidden, (2048, 2048, 2048))
        with self.assertRaises(ValueError):
            RunConfig(replay_capacity=8)

    def test_full_clip_evaluation_padding_and_early_failure(self):
        class Data(NamedTuple):
            qpos: object

        class State(NamedTuple):
            episode_steps: object
            motion_ids: object
            data: object
            observation_history: object

        class Env:
            num_envs = 2
            domain_randomization = observation_noise = False
            dt = 1.
            reference = SimpleNamespace(fps=1., lengths=jnp.array([6, 4, 3]),
                                        num_motions=3, motion_names=("long", "fails", "short"))
            state = State(jnp.zeros(2, jnp.int32), jnp.zeros(2, jnp.int32),
                          Data(jnp.zeros((2, 8))), jnp.zeros((2, 1, 1)))

            def reset_jax(self, state, mask=None, motion_ids=None, frames=None):
                mask = jnp.ones(2, bool) if mask is None else mask
                return state._replace(episode_steps=jnp.where(mask, 0, state.episode_steps),
                                      motion_ids=jnp.where(mask, motion_ids, state.motion_ids)), None, {}

            def motion_goal(self, state, count):
                return jnp.zeros((2, count, 1))

            def _reference(self, state):
                return jnp.zeros((2, 8)), None

            def step_jax(self, state, action):
                following = state._replace(episode_steps=state.episode_steps + 1)
                failed = (state.motion_ids == 1) & (following.episode_steps == 1)
                truncated = following.episode_steps >= self.reference.lengths[state.motion_ids] - 1
                return following, None, jnp.full(2, 5.), failed, truncated, {}

        actor = SimpleNamespace(config=SimpleNamespace(future_frames=1),
                                apply=lambda variables, observation, goals: jnp.zeros((2, 1)))
        env = Env()
        env.max_episode_steps = 1
        result = Evaluator(env, actor).evaluate({})
        self.assertEqual(result["failures"], 1)
        self.assertEqual([row["steps"] for row in result["clips"]], [5, 1, 2])
        np.testing.assert_allclose([row["normalized_return"] for row in result["clips"]], [1., 1/3, 1.])


if __name__ == "__main__":
    unittest.main()
