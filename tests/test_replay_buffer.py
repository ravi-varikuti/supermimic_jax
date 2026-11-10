"""Run with: JAX_PLATFORMS=cpu python -m unittest discover -s tests -v"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from replay_buffer import ReplayBuffer
from robot_config import RobotConfig
from transitions import Transition


CONFIG = RobotConfig(state_shapes=(("x", (2,)),), context_shapes=(),
                     action_dim=1, observation_dim=1, history_length=4)


def record(time, steps, terminated=(False, False), truncated=(False, False)):
    value = jnp.array([[time], [100 + time]], jnp.float32)
    return Transition(
        state=jnp.repeat(value, 2, axis=-1), action=jnp.zeros((2, 1)),
        next_state=jnp.repeat(value + 1, 2, axis=-1), reward=value[:, 0],
        terminated=jnp.array(terminated), truncated=jnp.array(truncated),
        context=jnp.zeros((2, 0)), previous_action=jnp.zeros((2, 1)),
        observation=value, next_observation=value + 1,
        metadata={"episode_steps": jnp.array(steps), "time": jnp.full(2, time)},
    )


class ReplayTests(unittest.TestCase):
    def test_empty_and_not_enough_steps(self):
        replay = ReplayBuffer(CONFIG, capacity=6, sequence_length=3)
        state = replay.init(record(0, [0, 0]))
        for _ in range(3):
            sample = jax.jit(lambda s: replay.sample(s, jax.random.key(0), 4))(state)
            self.assertFalse(bool(sample.valid))
            np.testing.assert_array_equal(sample.transitions.state, 0)
            np.testing.assert_array_equal(sample.history, 0)
            state = replay.add(state, record(int(state.size), [int(state.size)] * 2))

    def test_terminal_final_step_and_reset_padding(self):
        replay = ReplayBuffer(CONFIG, capacity=6, sequence_length=3)
        state = replay.init(record(0, [0, 0]))
        for t in range(3):
            state = replay.add(state, record(t, [t, t], terminated=(t == 2, False)))
        self.assertEqual(int(replay.valid_starts(state).sum()), 2)
        sample = replay.sample(state, jax.random.key(1), 32)
        self.assertTrue(bool(sample.valid))
        np.testing.assert_array_equal(sample.transitions.metadata["time"], np.tile([0, 1, 2], (32, 1)))
        np.testing.assert_array_equal(sample.history, np.repeat(sample.history[:, -1:], 4, axis=1))
        self.assertFalse(bool(sample.transitions.done[:, :-1].any()))

    def test_wraparound_history_and_episode_boundaries(self):
        replay = ReplayBuffer(CONFIG, capacity=6, sequence_length=3)
        state = replay.init(record(0, [0, 0]))
        records = []
        for t in range(10):
            # World 0 resets every 3 steps, world 1 every 5 steps.
            item = record(t, [t % 3, t % 5], terminated=(t % 3 == 2, False),
                          truncated=(False, t % 5 == 4))
            records.append(item)
            state = replay.add(state, item)
        self.assertEqual(int(state.size), 6)
        self.assertEqual(int(state.write_index), 4)
        # Retained rows are times 4..9. Enumerate eligibility independently.
        expected = np.zeros((6, 2), bool)
        for i, t in enumerate(range(4, 10)):
            for world, period in enumerate([3, 5]):
                expected[i, world] = (t + 3 <= 10 and t % period + 3 <= period
                                      and t - min(t % period, 3) >= 4)
        np.testing.assert_array_equal(replay.valid_starts(state), expected)
        sample = jax.jit(lambda s, k: replay.sample(s, k, 128))(state, jax.random.key(2))
        self.assertTrue(bool(sample.valid))
        self.assertEqual(int(sample.num_valid), int(expected.sum()))
        self.assertEqual(sample.transitions.state.shape, (128, 3, 2))
        self.assertEqual(sample.transitions.context.shape, (128, 3, 0))
        self.assertFalse(bool(sample.transitions.done[:, :-1].any()))
        times = np.array(sample.transitions.metadata["time"])
        np.testing.assert_array_equal(np.diff(times, axis=1), 1)
        for i in range(128):
            t = int(times[i, 0])
            world = int(sample.transitions.observation[i, 0, 0] >= 100)
            period = [3, 5][world]
            reset_time = t - t % period
            history_times = np.maximum(np.arange(t - 3, t + 1), reset_time)
            np.testing.assert_array_equal(sample.history[i, :, 0], history_times + 100 * world)
        again = replay.sample(state, jax.random.key(2), 128)
        np.testing.assert_array_equal(sample.transitions.state, again.transitions.state)

    def test_extend_matches_add_and_retains_latest_rows(self):
        replay = ReplayBuffer(CONFIG, capacity=5, sequence_length=2)
        items = [record(t, [t, t]) for t in range(12)]
        initial = replay.init(items[0])
        collection = jax.tree.map(lambda *values: jnp.stack(values), *items)
        extended = jax.jit(replay.extend)(initial, collection)
        added = initial
        for item in items:
            added = replay.add(added, item)
        for actual, expected in zip(jax.tree.leaves(extended), jax.tree.leaves(added)):
            np.testing.assert_array_equal(actual, expected)
        oldest = (int(extended.write_index) - int(extended.size)) % 5
        slots = (oldest + np.arange(5)) % 5
        np.testing.assert_array_equal(extended.data.metadata["time"][slots, 0], np.arange(7, 12))
        self.assertEqual(int(initial.size), 0)  # Original state was not mutated.

    def test_manual_reset_and_missing_history(self):
        replay = ReplayBuffer(CONFIG, capacity=8, sequence_length=2)
        state = replay.init(record(0, [7, 7]))
        for t, steps in enumerate([7, 8, 0, 1, 2, 3]):
            state = replay.add(state, record(t, [steps, steps]))
        expected = np.zeros((8, 2), bool)
        expected[2:5] = True
        np.testing.assert_array_equal(replay.valid_starts(state), expected)

    def test_derived_episode_steps_without_metadata(self):
        replay = ReplayBuffer(CONFIG, capacity=4, sequence_length=1)
        first = record(0, [0, 0], terminated=(True, False))._replace(metadata={})
        state = replay.add(replay.init(first), first)
        second = record(1, [0, 0])._replace(metadata={})
        state = replay.add(state, second)
        np.testing.assert_array_equal(state.episode_steps[:2], [[0, 0], [0, 1]])


if __name__ == "__main__":
    unittest.main()
