"""Paper tracking rewards from predicted states using differentiable kinematics."""

from math import isfinite

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from normalization import prepare_state
import rewards
from state_representation import unpack_state


class TrackingRewardModel:
    def __init__(self, native_model, config, body_names, anchor_body, dt):
        """Configure a robot's tracked bodies; use the same Table A.2 reward math.

        Supports qpos/qvel state fields and one hinge/slide joint per actuator.
        Kinematics uses nominal inertias: body-origin poses/velocities depend on
        qpos/qvel and geometry, not the episode's friction, mass, or COM offsets.
        """
        shapes = dict(config.state_shapes)
        if shapes.get("qpos") != (native_model.nq,) or shapes.get("qvel") != (native_model.nv,):
            raise ValueError("State config must contain matching qpos/qvel fields")
        if not isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        body_names = tuple(body_names)
        if not body_names or len(set(body_names)) != len(body_names) or anchor_body not in body_names:
            raise ValueError("Tracked bodies must be unique and contain the anchor")
        joints = native_model.actuator_trnid[:, 0]
        if config.action_dim != native_model.nu or len(set(joints.tolist())) != native_model.nu:
            raise ValueError("Expected one action/actuator per distinct joint")
        if np.any(native_model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError("Tracking reward expects joint actuators")
        if not np.isin(native_model.jnt_type[joints], [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]).all():
            raise ValueError("Joint-limit reward expects hinge or slide joints")
        roots = [int(native_model.body_rootid[native_model.body(name).id]) for name in body_names]
        if len(set(roots)) != 1:
            raise ValueError("Tracked bodies must belong to one robot root")
        self.config, self.dt = config, float(dt)
        self.anchor_index = body_names.index(anchor_body)
        self.joint_limits = rewards.soft_joint_limits(native_model)
        self.model = mjx.put_model(native_model, impl="jax")
        template = mjx.make_data(self.model)

        def kinematics(model, qpos, qvel):
            data = template.replace(qpos=qpos, qvel=qvel)
            data = mjx.kinematics(model, data)
            data = mjx.com_pos(model, data)
            data = mjx.com_vel(model, data)
            return rewards.from_data(native_model, data, body_names)

        self._kinematics = jax.jit(jax.vmap(kinematics, in_axes=(None, 0, 0)))

    def tracking_state(self, state):
        """Compute body-origin poses and world velocities; no contacts or stepping."""
        state = jnp.asarray(state, jnp.float32)
        fields = unpack_state(prepare_state(state, self.config), self.config)
        batch = state.shape[:-1]
        tracking = self._kinematics(self.model, fields["qpos"].reshape(-1, self.model.nq),
                                    fields["qvel"].reshape(-1, self.model.nv))
        return jax.tree.map(lambda value: value.reshape(batch + value.shape[1:]), tracking)

    def __call__(self, next_state, reference_state, action, previous_action):
        """Return (total, weighted_terms), preserving leading batch/time axes."""
        next_state, reference_state = jnp.asarray(next_state, jnp.float32), jnp.asarray(reference_state, jnp.float32)
        action, previous_action = jnp.asarray(action, jnp.float32), jnp.asarray(previous_action, jnp.float32)
        if next_state.shape != reference_state.shape:
            raise ValueError("Predicted and reference states must have identical shapes")
        shape = next_state.shape[:-1] + (self.config.action_dim,)
        if action.shape != shape or previous_action.shape != shape:
            raise ValueError(f"Actions must have shape {shape}")
        return rewards.tracking_reward(
            self.tracking_state(next_state), self.tracking_state(reference_state),
            action, previous_action, self.joint_limits, self.dt, self.anchor_index,
        )
