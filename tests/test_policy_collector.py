from typing import NamedTuple
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from collector import CollectorState, PolicyCollector
from configs.g1 import G1_LEGACY_CONFIG as G1_CONFIG
from policy import MotionPolicy, PolicyConfig
from reference import ReferenceMotion
from rl_env import G1Env
from robot_config import RobotConfig


ROBOT = RobotConfig(state_shapes=(("x", (2,)),), context_shapes=(),
                    action_dim=1, observation_dim=2, history_length=2)


class FakeState(NamedTuple):
    position: object
    previous_action: object
    episode_steps: object
    observation_history: object
    clean_observation_history: object


class FakeEnv:
    robot_config = ROBOT
    num_envs = 2

    def initial(self):
        return FakeState(jnp.zeros((2, 2)), jnp.zeros((2, 1)), jnp.zeros(2, jnp.int32),
                         jnp.zeros((2, 2, 2)), jnp.zeros((2, 2, 2)))

    def get_state(self, state):
        return {"x": state.position, "previous_action": state.previous_action,
                "clean_observation_history": state.clean_observation_history,
                "domain_randomization": {}, "episode_steps": state.episode_steps}

    def step_jax(self, state, action):
        position = state.position + action
        history = jnp.concatenate((state.observation_history[:, 1:], position[:, None]), axis=1)
        state = FakeState(position, action, state.episode_steps + 1, history, history)
        return state, history.reshape(2, 4), action[:, 0], state.episode_steps >= 2, jnp.zeros(2, bool), {}

    def reset_jax(self, state, mask):
        state = FakeState(
            jnp.where(mask[:, None], 0., state.position),
            jnp.where(mask[:, None], 0., state.previous_action),
            jnp.where(mask, 0, state.episode_steps),
            jnp.where(mask[:, None, None], 0., state.observation_history),
            jnp.where(mask[:, None, None], 0., state.clean_observation_history),
        )
        return state, state.observation_history.reshape(2, 4), {}


class ConstantPolicy:
    robot_config = ROBOT

    def apply(self, variables, observation, goal):
        return jnp.full((2, 1), variables["action"])


class PolicyTests(unittest.TestCase):
    def test_default_policy_shapes(self):
        policy = MotionPolicy(G1_CONFIG)
        obs, goal = jnp.zeros((2, 930)), jnp.zeros((2, 10, 71))
        variables = policy.init(jax.random.key(0), obs, goal)
        result = jax.jit(policy.apply)(variables, obs, goal)
        self.assertEqual(result.shape, (2, 29))
        self.assertTrue(bool(jnp.isfinite(result).all() & (jnp.abs(result) <= 1.).all()))

    def test_other_robot_and_custom_goal_gradients(self):
        config = PolicyConfig(future_frames=3, goal_frame_dim=4, latent_dim=5,
                              encoder_hidden=(8,), decoder_hidden=(8,))
        policy = MotionPolicy(ROBOT, config)
        obs, goal = jnp.ones((2, 4)), jnp.ones((2, 3, 4))
        variables = policy.init(jax.random.key(1), obs, goal)
        gradients = jax.grad(lambda o, g: policy.apply(variables, o, g).sum(),
                             argnums=(0, 1))(obs, goal)
        for gradient in gradients:
            self.assertTrue(bool(jnp.isfinite(gradient).all()))
            self.assertGreater(float(jnp.linalg.norm(gradient)), 0.)
        with self.assertRaises(ValueError):
            policy.apply(variables, obs, jnp.ones((2, 4, 4)))

    def test_clipping_and_terminal_records_precede_reset(self):
        env = FakeEnv()
        collector = PolicyCollector(env, ConstantPolicy(), lambda s: jnp.zeros((2, 1, 2)),
                                    lambda b, a: {"episode_steps": b["episode_steps"]}, exploration_std=0.)
        carry = CollectorState(env.initial(), jax.random.key(2))
        result, records = jax.jit(lambda c: collector.collect({"action": 2.}, c, 4))(carry)
        np.testing.assert_array_equal(records.action, 1.)
        np.testing.assert_array_equal(records.state[:, 0, 0], [0., 1., 0., 1.])
        np.testing.assert_array_equal(records.next_state[:, 0, 0], [1., 2., 1., 2.])
        np.testing.assert_array_equal(records.terminated[:, 0], [False, True, False, True])
        np.testing.assert_array_equal(records.metadata["episode_steps"][:, 0], [0, 1, 0, 1])
        np.testing.assert_array_equal(result.env_state.position, 0.)
        np.testing.assert_array_equal(carry.env_state.position, 0.)

    def test_exploration_rng_and_no_noise_evaluation(self):
        env, key = FakeEnv(), jax.random.key(3)
        goal_fn = lambda state: jnp.zeros((2, 1, 2))
        collector = PolicyCollector(env, ConstantPolicy(), goal_fn, exploration_std=.1)
        collect = jax.jit(lambda c: collector.collect({"action": 0.}, c, 3))
        carry = CollectorState(env.initial(), key)
        _, records = collect(carry)
        _, again = collect(carry)
        noise_key = jax.random.split(key)[1]
        np.testing.assert_allclose(records.action[0], .1 * jax.random.normal(noise_key, (2, 1)), atol=1e-7)
        np.testing.assert_array_equal(records.action, again.action)
        clean = PolicyCollector(env, ConstantPolicy(), goal_fn, exploration_std=0.)
        _, records = clean.collect({"action": .25}, carry, 3)
        np.testing.assert_array_equal(records.action, .25)

    def test_future_goal_reconstruction_and_clamping(self):
        env = G1Env.__new__(G1Env)
        env.robot_config, env.dt = G1_CONFIG, .015
        reference = ReferenceMotion.__new__(ReferenceMotion)
        reference.fps = 50.
        reference.lengths, reference.offsets = jnp.array([7, 7]), jnp.array([0, 7])
        qpos = jnp.zeros((14, 36)).at[:, 3].set(1.).at[:, 0].set(jnp.tile(jnp.arange(7), 2))
        reference.qpos, reference.qvel = qpos, jnp.zeros((14, 35))
        env.reference = reference
        metadata = {"episode_steps": jnp.array([1, 2]), "reference_frame": jnp.array([0, 5]),
                    "motion_id": jnp.array([0, 1]), "reference_origin": jnp.zeros((2, 3))}
        goals = jax.jit(lambda m: env.motion_goal_from_metadata(m, 3))(metadata)
        self.assertEqual(goals.shape, (2, 3, 71))
        np.testing.assert_array_equal(goals[..., 0], [[1., 2., 3.], [6., 6., 6.]])
        np.testing.assert_array_equal(goals[..., 3], 1.)

        # Imagination extends independently of replay window length, at the same
        # noninteger control/reference rate, and bootstraps at episode boundaries.
        from types import SimpleNamespace
        from replay_buffer import ReplaySample
        env.max_episode_steps = 4
        transitions = SimpleNamespace(
            metadata=jax.tree.map(lambda x: x[:, None], metadata),
            state=jnp.zeros((2, 1, 71)), context=jnp.zeros((2, 1, 33)),
            previous_action=jnp.zeros((2, 1, 29)),
        )
        sample = ReplaySample(transitions, jnp.zeros((2, 10, 93)), jnp.array(True), jnp.array(2))
        batch = jax.jit(lambda: env.imagination_batch(sample, horizon=4, goal_frames=3))()
        self.assertEqual(batch.goals.shape, (2, 5, 3, 71))
        np.testing.assert_array_equal(batch.goals[:, 0], goals)
        np.testing.assert_array_equal(batch.reference[..., 0], [[1., 2., 3., 3.], [6., 6., 6., 6.]])
        np.testing.assert_array_equal(batch.truncated, [[False, False, True, True], [True, True, True, True]])


if __name__ == "__main__":
    unittest.main()
