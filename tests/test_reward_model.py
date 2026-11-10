import copy
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from configs.g1 import G1_CONFIG
from reward_model import TrackingRewardModel
import rewards
from robot_config import RobotConfig
from state_representation import pack_state


class RewardFormulaTests(unittest.TestCase):
    def test_table_a2_terms_and_timestep(self):
        count = len(rewards.TRACKED_BODIES)
        zero = jnp.zeros((count, 3))
        identity = jnp.tile(jnp.array([1., 0., 0., 0.]), (count, 1))
        reference = rewards.TrackingState(zero, identity, zero, zero, jnp.zeros(29))
        angle = .2
        quaternion = jnp.tile(jnp.array([jnp.cos(angle / 2), jnp.sin(angle / 2), 0., 0.]), (count, 1))
        robot = rewards.TrackingState(
            zero.at[:, 2].set(.1), quaternion, zero.at[:, 0].set(.5),
            zero.at[:, 1].set(1.), jnp.zeros(29).at[:2].set(1.5),
        )
        limits = jnp.tile(jnp.array([-1., 1.]), (29, 1))
        for dt in (.01, .02):
            total, terms = rewards.tracking_reward(robot, reference, jnp.full(29, .2),
                                                   jnp.zeros(29), limits, dt)
            expected = {
                "body_position": np.exp(-.1 ** 2 / .3 ** 2),
                "body_orientation": np.exp(-angle ** 2 / .4 ** 2),
                "body_linear_velocity": np.exp(-.5 ** 2),
                "body_angular_velocity": np.exp(-1. / 3.14 ** 2),
                "root_position": .5 * np.exp(-.1 ** 2 / .3 ** 2),
                "root_orientation": .5 * np.exp(-angle ** 2 / .4 ** 2),
                "action_rate": -.1 * 29 * .2 ** 2,
                "joint_limit": -2.,
            }
            self.assertEqual(set(terms), set(expected))
            for name, value in expected.items():
                np.testing.assert_allclose(terms[name], value * dt, rtol=2e-6, atol=1e-7)
            np.testing.assert_allclose(total, sum(expected.values()) * dt, rtol=2e-6)


class KinematicRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "unitree_g1/scene_29dof_mjlab.xml"
        cls.native = mujoco.MjModel.from_xml_path(str(path))
        cls.reward_model = TrackingRewardModel(cls.native, G1_CONFIG, rewards.TRACKED_BODIES,
                                               "torso_link", .02)
        qpos = cls.native.qpos0.copy()
        qpos[2] = 1.5
        quaternion = np.array([.8, .1, -.2, .3])
        qpos[3:7] = quaternion / np.linalg.norm(quaternion)
        joint_ids = cls.native.actuator_trnid[:, 0]
        qpos[cls.native.jnt_qposadr[joint_ids]] = cls.native.jnt_range[joint_ids].mean(axis=-1)
        cls.qpos = qpos
        cls.qvel = np.linspace(-.3, .4, cls.native.nv)

    def native_state(self, qpos, qvel, model=None):
        model = self.native if model is None else model
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        return data

    def test_body_poses_and_velocities_match_native_with_com_offset(self):
        perturbed = copy.copy(self.native)
        perturbed.body_ipos[perturbed.body("torso_link").id] += [.02, -.03, .01]
        native = self.native_state(self.qpos, self.qvel, perturbed)
        packed = pack_state({"qpos": self.qpos, "qvel": self.qvel}, G1_CONFIG)
        actual = self.reward_model.tracking_state(packed)
        for i, name in enumerate(rewards.TRACKED_BODIES):
            body = perturbed.body(name).id
            np.testing.assert_allclose(actual.body_pos[i], native.xpos[body], atol=3e-6)
            # Quaternion sign is equivalent; compare absolute inner product.
            np.testing.assert_allclose(abs(np.dot(np.array(actual.body_quat[i]), native.xquat[body])), 1., atol=1e-6)
            jp, jr = np.zeros((3, perturbed.nv)), np.zeros((3, perturbed.nv))
            mujoco.mj_jacBody(perturbed, native, jp, jr, body)
            np.testing.assert_allclose(actual.body_lin_vel[i], jp @ self.qvel, atol=3e-6)
            np.testing.assert_allclose(actual.body_ang_vel[i], jr @ self.qvel, atol=3e-6)

    def test_reward_matches_native_and_has_finite_gradients(self):
        reference = self.native_state(self.qpos, self.qvel)
        changed = self.qpos.copy()
        changed[2] += .03
        changed[7] += .05
        robot = self.native_state(changed, self.qvel + .02)
        state = pack_state({"qpos": changed, "qvel": self.qvel + .02}, G1_CONFIG)
        target = pack_state({"qpos": self.qpos, "qvel": self.qvel}, G1_CONFIG)
        action, previous = jnp.full(29, .1), jnp.zeros(29)
        expected, expected_terms = rewards.tracking_reward(
            rewards.from_data(self.native, robot), rewards.from_data(self.native, reference),
            action, previous, self.reward_model.joint_limits, .02,
        )
        actual, terms = self.reward_model(state, target, action, previous)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        for name in terms:
            np.testing.assert_allclose(terms[name], expected_terms[name], rtol=1e-5, atol=1e-6)
        grad_fn = jax.jit(jax.grad(lambda s, a: self.reward_model(s, target, a, previous)[0], argnums=(0, 1)))
        gradients = grad_fn(state, action)
        for gradient in gradients:
            self.assertTrue(bool(jnp.isfinite(gradient).all()))
            self.assertGreater(float(jnp.linalg.norm(gradient)), 0.)
        gradient = grad_fn(target, previous)[0]
        self.assertTrue(bool(jnp.isfinite(gradient).all()))

    def test_batch_and_time_axes(self):
        state = pack_state({"qpos": self.qpos, "qvel": self.qvel}, G1_CONFIG)
        batch = jnp.broadcast_to(state, (2, 3, G1_CONFIG.state_dim))
        action = jnp.zeros((2, 3, 29))
        reward, terms = self.reward_model(batch, batch, action, action)
        self.assertEqual(reward.shape, (2, 3))
        np.testing.assert_allclose(reward, .1, atol=1e-6)
        np.testing.assert_allclose(reward, sum(terms.values()), atol=1e-7)

    def test_other_robot_body_names_and_anchor(self):
        native = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="root" pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/>
            <body name="link" pos="0 0 .2"><joint name="hinge" type="hinge" range="-1 1"/>
              <geom type="capsule" size=".03" fromto="0 0 0 0 0 .2"/>
            </body></body></worldbody><actuator><position joint="hinge" kp="10"/>
          </actuator></mujoco>''')
        config = RobotConfig(state_shapes=(("qpos", (8,)), ("qvel", (7,))),
                             context_shapes=(), action_dim=1, observation_dim=4,
                             quaternion_slices=((3, 7),))
        model = TrackingRewardModel(native, config, ("root", "link"), "root", .02)
        state = pack_state({"qpos": native.qpos0, "qvel": jnp.zeros(7)}, config)
        reward, _ = model(state, state, jnp.zeros(1), jnp.zeros(1))
        np.testing.assert_allclose(reward, .1, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
