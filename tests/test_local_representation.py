import unittest
from pathlib import Path

from flax.core import unfreeze
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from configs.g1 import G1_CONFIG, controller_defaults
from geometry import (conjugate, exp_rotation, heading, integrate_local, invariant_state,
                      local_delta, log_rotation, multiply, rotate, rotation6d)
from normalization import FeatureStats, Normalization
from policy import MotionPolicy, PolicyConfig
from reference import ReferenceMotion
from rl_env import G1Env
from running_norm import empty_moments, update_moments
from world_model import WorldModel, WorldModelConfig
import observations


def physical_states(count=3):
    state = jax.random.normal(jax.random.key(1), (count, 71)) * .1
    return state.at[:, 2].set(.8).at[:, 3:7].set(exp_rotation(jnp.array([.1, .2, .3])))


def transform(state):
    yaw = exp_rotation(jnp.array([0., 0., 1.2]))
    return state.at[..., :3].set(rotate(yaw, state[..., :3]) + jnp.array([4., -2., 0.])).at[..., 3:7].set(
        multiply(yaw, state[..., 3:7])).at[..., 36:39].set(rotate(yaw, state[..., 36:39]))


class LocalRepresentationTests(unittest.TestCase):
    def test_controller_pose_and_motor_scaling(self):
        model = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1] / 'unitree_g1/scene_29dof_mjlab.xml'))
        nominal, scale = controller_defaults(model)
        for joint, expected_pose, expected_scale in [('left_knee_joint', .669, .35066147),
                                                    ('left_hip_pitch_joint', -.312, .54754646),
                                                    ('left_elbow_joint', .6, .43857731)]:
            index = int(np.flatnonzero(model.actuator_trnid[:, 0] == model.joint(joint).id)[0])
            self.assertAlmostEqual(float(nominal[index]), expected_pose, places=6)
            self.assertAlmostEqual(float(scale[index]), expected_scale, places=6)

    def test_local_integration_roundtrip_and_finite_zero_rotation_gradients(self):
        state = physical_states()
        delta = jax.random.normal(jax.random.key(2), (3, 70)) * .02
        following = integrate_local(state, delta)
        np.testing.assert_allclose(local_delta(state, following), delta, atol=2e-7)
        np.testing.assert_allclose(integrate_local(state, jnp.zeros_like(delta)), state, atol=2e-7)
        np.testing.assert_allclose(jnp.linalg.norm(following[:, 3:7], axis=-1), 1., atol=1e-6)
        gradient = jax.jacrev(lambda x: log_rotation(exp_rotation(x)))(jnp.zeros(3))
        np.testing.assert_allclose(gradient, jnp.eye(3), atol=1e-6)

    def test_heading_translation_invariance_and_equivariance(self):
        state = physical_states()
        delta = jax.random.normal(jax.random.key(3), (3, 70)) * .03
        np.testing.assert_allclose(invariant_state(transform(state)), invariant_state(state), atol=1e-6)
        np.testing.assert_allclose(integrate_local(transform(state), delta),
                                   transform(integrate_local(state, delta)), atol=1e-6)

    def test_reference_relative_observations(self):
        state = physical_states()
        target = state.at[:, 3:7].set(exp_rotation(jnp.array([0., 0., 1.])))
        obs = observations.proprioception(state[:, :36], state[:, 36:], jnp.zeros((3, 29)),
                                          jnp.zeros(29), reference_qpos=target[:, :36])
        moved, moved_ref = transform(state), transform(target)
        after = observations.proprioception(moved[:, :36], moved[:, 36:], jnp.zeros((3, 29)),
                                            jnp.zeros(29), reference_qpos=moved_ref[:, :36])
        self.assertEqual(obs.shape, (3, 99))
        np.testing.assert_allclose(obs, after, atol=1e-6)
        self.assertEqual(observations.noise_scale(True).shape, (99,))

    def test_goal_features_are_heading_invariant_and_origin_independent(self):
        env = G1Env.__new__(G1Env)
        env.robot_config, env.dt = G1_CONFIG, .02
        ref = ReferenceMotion.__new__(ReferenceMotion)
        ref.fps, ref.lengths, ref.offsets = 50., jnp.array([8]), jnp.array([0])
        states = physical_states(8).at[:, 0].set(jnp.arange(8) * .02).at[:, 1].set(0.)
        ref.qpos, ref.qvel = states[:, :36], states[:, 36:]
        env.reference = ref
        metadata = {'motion_id': jnp.array([0]), 'reference_frame': jnp.array([1]),
                    'episode_steps': jnp.array([1]), 'reference_origin': jnp.array([[3., 4., 0.]])}
        goal = env.motion_goal_from_metadata(metadata, 3)
        self.assertEqual(goal.shape, (1, 3, 73))
        np.testing.assert_allclose(goal[0, :, :3], rotate(conjugate(heading(states[1, 3:7])),
            jnp.array([[.02, 0., 0.], [.04, 0., 0.], [.06, 0., 0.]])), atol=1e-6)
        moved = transform(states)
        ref.qpos, ref.qvel = moved[:, :36], moved[:, 36:]
        np.testing.assert_allclose(env.motion_goal_from_metadata(metadata, 3), goal, atol=1e-6)

    def test_running_moments_match_full_data_with_missing_rows(self):
        values = jax.random.normal(jax.random.key(4), (20, 5))
        direct = update_moments(empty_moments(5), values)
        staged = update_moments(update_moments(empty_moments(5), values[:7]), values[7:])
        np.testing.assert_allclose(staged.mean, direct.mean, atol=1e-6)
        np.testing.assert_allclose(staged.variance, direct.variance, atol=1e-6)
        unchanged = update_moments(staged, jnp.full((2, 5), jnp.nan))
        np.testing.assert_array_equal(unchanged.mean, staged.mean)
        self.assertEqual(float(unchanged.count), 20.)

    def test_linear_actor_can_command_outside_old_range(self):
        policy = MotionPolicy(G1_CONFIG, PolicyConfig(goal_frame_dim=73, bound_actions=False,
                                                      encoder_hidden=(8,), decoder_hidden=(8,)))
        obs, goal = jnp.zeros((1, 990)), jnp.zeros((1, 10, 73))
        params = unfreeze(policy.init(jax.random.key(5), obs, goal)['params'])
        params['decoder']['output']['bias'] = jnp.full(29, 2.)
        action = policy.apply({'params': params}, obs, goal)
        np.testing.assert_allclose(action, 2.)

    def test_local_model_uses_rotation_composition_and_ignores_world_origin(self):
        model = WorldModel(G1_CONFIG, WorldModelConfig(hidden_dims=(8,), local_frame=True))
        state = physical_states()
        action, context = jnp.ones((3, 29)), jnp.zeros((3, 33))
        stats = Normalization(FeatureStats(jnp.zeros(71), jnp.ones(71)),
                              FeatureStats(jnp.zeros(33), jnp.ones(33)),
                              FeatureStats(jnp.zeros(70), jnp.ones(70)), jnp.array(1.))
        params = unfreeze(model.init(jax.random.key(6), state, action, context, stats)['params'])
        params['dynamics']['output']['bias'] = jnp.linspace(-.01, .01, 70)
        predicted = model.apply({'params': params}, state, action, context, stats)
        moved = model.apply({'params': params}, transform(state), action, context, stats)
        np.testing.assert_allclose(moved, transform(predicted), atol=1e-6)
        gradient = jax.grad(lambda a: model.apply({'params': params}, state, a, context, stats).sum())(action)
        self.assertTrue(bool(jnp.isfinite(gradient).all()))


if __name__ == '__main__':
    unittest.main()
