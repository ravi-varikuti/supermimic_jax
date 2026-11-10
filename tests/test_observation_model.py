import unittest

import jax
import jax.numpy as jnp
import numpy as np

from configs.g1 import G1_LEGACY_CONFIG as G1_CONFIG
from normalization import unit_quaternion
from observation_model import ObservationModel
import observations
from rl_env import G1Env
from robot_config import RobotConfig
from state_representation import pack_context, pack_state


ROBOT = RobotConfig(state_shapes=(("x", (3,)),), context_shapes=(("bias", ()),),
                    action_dim=2, observation_dim=5, history_length=3)


def adapter(state, action, context):
    return jnp.concatenate((state + context, action), axis=-1)


class ObservationTests(unittest.TestCase):
    def test_g1_adapter_matches_simulation_formula(self):
        env = G1Env.__new__(G1Env)
        env.robot_config = G1_CONFIG
        env.default_joint_pos = jnp.linspace(-.2, .2, 29)
        qpos = jnp.zeros((2, 36)).at[:, 3:7].set(unit_quaternion(jnp.array([.8, .2, .1, .3])))
        qpos = qpos.at[:, 7:].set(jnp.linspace(-.5, .5, 29))
        qvel, action = jnp.arange(70, dtype=jnp.float32).reshape(2, 35) / 10., jnp.ones((2, 29)) * .3
        context = {"friction": jnp.full(2, .6), "joint_bias": jnp.full((2, 29), .01),
                   "torso_com_offset": jnp.zeros((2, 3))}
        packed = pack_state({"qpos": qpos, "qvel": qvel}, G1_CONFIG)
        parameters = pack_context(context, G1_CONFIG)
        model = ObservationModel(G1_CONFIG, env.observation_from_state, observations.noise_scale())
        actual = jax.jit(model.clean)(packed, action, parameters)
        expected = observations.proprioception(qpos, qvel, action, env.default_joint_pos, context["joint_bias"])
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        np.testing.assert_array_equal(actual[:, -29:], action)
        gradient = jax.grad(lambda state: model.clean(state, action, parameters).sum())(packed)
        self.assertTrue(bool(jnp.isfinite(gradient).all()))
        self.assertGreater(float(jnp.linalg.norm(gradient)), 0.)

    def test_history_shift_noise_and_actions(self):
        scale = jnp.array([.1, .2, .3, 0., 0.])
        model = ObservationModel(ROBOT, adapter, scale)
        initial = jnp.arange(30, dtype=jnp.float32).reshape(2, 3, 5)
        history = model.init_history(initial, jax.random.key(0))
        state, action, context = jnp.ones((2, 3)), jnp.full((2, 2), .5), jnp.full((2, 1), .1)
        following = jax.jit(model.step)(history, state, action, context)
        np.testing.assert_array_equal(following.clean[:, :-1], history.clean[:, 1:])
        np.testing.assert_array_equal(following.noisy[:, :-1], history.noisy[:, 1:])
        np.testing.assert_allclose(following.clean[:, -1], adapter(state, action, context))
        difference = np.asarray(following.noisy[:, -1] - following.clean[:, -1])
        self.assertTrue(np.all(np.abs(difference) <= np.array(scale) + 1e-6))
        np.testing.assert_array_equal(following.noisy[:, -1, -2:], action)
        self.assertEqual(following.observation.shape, (2, 15))
        np.testing.assert_array_equal(following.observation, following.noisy.reshape(2, 15))
        again = model.step(history, state, action, context)
        np.testing.assert_allclose(following.noisy, again.noisy, atol=1e-6)

    def test_reset_padding_shares_initial_noise(self):
        model = ObservationModel(ROBOT, adapter, jnp.array([.1, .2, .3, 0., 0.]))
        clean = jnp.zeros((3, 3, 5))
        history = model.init_history(clean, jax.random.key(1), episode_steps=jnp.array([0, 1, 2]))
        np.testing.assert_array_equal(history.noisy[0], jnp.repeat(history.noisy[0, :1], 3, axis=0))
        np.testing.assert_array_equal(history.noisy[1, 0], history.noisy[1, 1])
        self.assertFalse(np.array_equal(history.noisy[1, 1], history.noisy[1, 2]))
        self.assertFalse(np.array_equal(history.noisy[2, 0], history.noisy[2, 1]))

    def test_noise_disabled_and_scan_gradients(self):
        model = ObservationModel(ROBOT, adapter, jnp.ones(5), noise=False)
        history = model.init_history(jnp.zeros((2, 3, 5)), jax.random.key(2))
        actions, contexts = jnp.zeros((4, 2, 2)), jnp.zeros((4, 2, 1))

        def rollout(states):
            def advance(carry, values):
                following = model.step(carry, *values)
                return following, following.observation
            return jax.lax.scan(advance, history, (states, actions, contexts))

        states = jnp.ones((4, 2, 3))
        final, observations_out = jax.jit(rollout)(states)
        self.assertEqual(observations_out.shape, (4, 2, 15))
        np.testing.assert_array_equal(final.clean, final.noisy)
        gradient = jax.grad(lambda state: rollout(state)[1].sum())(states)
        self.assertTrue(bool(jnp.isfinite(gradient).all()))
        self.assertGreater(float(jnp.linalg.norm(gradient)), 0.)

    def test_invalid_shapes(self):
        model = ObservationModel(ROBOT, adapter)
        with self.assertRaises(ValueError):
            model.init_history(jnp.zeros((2, 2, 5)), jax.random.key(0))
        with self.assertRaises(ValueError):
            model.clean(jnp.zeros((2, 3)), jnp.zeros((2, 1)), jnp.zeros((2, 1)))


if __name__ == "__main__":
    unittest.main()
