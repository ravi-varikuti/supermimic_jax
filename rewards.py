"""SuperMimic tracking reward, following the pinned mjlab/BeyondMimic task.

Positive tracking terms sum to at most 5 * dt per control step. Every term,
including penalties, is scaled by dt. Physics self-collision has no reward penalty.
"""

from typing import NamedTuple

import jax.numpy as jnp

TRACKED_BODIES = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
ANCHOR_INDEX = TRACKED_BODIES.index("torso_link")


class TrackingState(NamedTuple):
    body_pos: object
    body_quat: object
    body_lin_vel: object
    body_ang_vel: object
    joint_pos: object


def from_data(model, data, body_names=TRACKED_BODIES):
    """Extract JAX link states from MuJoCo/MJX data with current kinematics."""
    ids = jnp.array([model.body(name).id for name in body_names])
    root = int(model.body_rootid[model.body(body_names[0]).id])
    pos, quat = jnp.asarray(data.xpos)[..., ids, :], jnp.asarray(data.xquat)[..., ids, :]
    spatial = jnp.asarray(data.cvel)[..., ids, :]
    ang = spatial[..., :3]
    lin = spatial[..., 3:] + jnp.cross(ang, pos - data.subtree_com[..., root, None, :])
    joint_ids = model.actuator_trnid[:, 0]
    joint_qpos_ids = model.jnt_qposadr[joint_ids]
    return TrackingState(pos, quat, lin, ang, jnp.asarray(data.qpos)[..., joint_qpos_ids])


def soft_joint_limits(model, factor=0.9):
    limits = jnp.asarray(model.jnt_range[model.actuator_trnid[:, 0]])
    center = limits.mean(axis=-1)
    half = factor * (limits[:, 1] - limits[:, 0]) / 2
    return jnp.stack((center - half, center + half), axis=-1)


def _conjugate(q):
    return q * jnp.array([1, -1, -1, -1])


def _multiply(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return jnp.concatenate((aw * bw - jnp.sum(av * bv, axis=-1, keepdims=True),
                            aw * bv + bw * av + jnp.cross(av, bv)), axis=-1)


def _rotate(q, vector):
    cross = 2 * jnp.cross(q[..., 1:], vector)
    return vector + q[..., :1] * cross + jnp.cross(q[..., 1:], cross)


def _yaw(q):
    w, x, y, z = (q[..., i] for i in range(4))
    angle = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = jnp.zeros_like(angle)
    return jnp.stack((jnp.cos(angle / 2), zero, zero, jnp.sin(angle / 2)), axis=-1)


def _orientation_error_squared(a, b):
    relative = _multiply(a, _conjugate(b))
    vector_squared = jnp.sum(relative[..., 1:] ** 2, axis=-1)
    angle = 2 * jnp.arctan2(jnp.sqrt(jnp.maximum(vector_squared, 1e-12)),
                           jnp.abs(relative[..., 0]))
    # Keep the gradient finite at exactly matching orientations.
    return jnp.where(vector_squared < 1e-8, 4 * vector_squared, angle**2)


def tracking_reward(robot, reference, action, previous_action, joint_limits, dt,
                    anchor_index=ANCHOR_INDEX):
    """Return (total, weighted_terms), scaled by the control timestep in seconds.

    Root reward uses the torso anchor as in mjlab. Reference body poses are
    re-anchored in xy/yaw; reference height and roll/pitch remain unchanged.
    Link velocities are compared in the world frame.
    """
    a = anchor_index
    robot_anchor = robot.body_pos[..., a, :]
    ref_anchor = reference.body_pos[..., a, :]
    delta = _yaw(_multiply(robot.body_quat[..., a, :],
                           _conjugate(reference.body_quat[..., a, :])))
    origin = robot_anchor.at[..., 2].set(ref_anchor[..., 2])
    desired_pos = origin[..., None, :] + _rotate(
        delta[..., None, :], reference.body_pos - ref_anchor[..., None, :])
    desired_quat = _multiply(delta[..., None, :], reference.body_quat)

    def body_error(actual, desired):
        return jnp.mean(jnp.sum((actual - desired) ** 2, axis=-1), axis=-1)

    lower, upper = joint_limits[:, 0], joint_limits[:, 1]
    terms = {
        "body_position": jnp.exp(-body_error(robot.body_pos, desired_pos) / 0.3**2),
        "body_orientation": jnp.exp(-jnp.mean(_orientation_error_squared(robot.body_quat, desired_quat), axis=-1) / 0.4**2),
        "body_linear_velocity": jnp.exp(-body_error(robot.body_lin_vel, reference.body_lin_vel) / 1.0**2),
        "body_angular_velocity": jnp.exp(-body_error(robot.body_ang_vel, reference.body_ang_vel) / 3.14**2),
        "root_position": 0.5 * jnp.exp(-jnp.sum((robot_anchor - ref_anchor) ** 2, axis=-1) / 0.3**2),
        "root_orientation": 0.5 * jnp.exp(-_orientation_error_squared(robot.body_quat[..., a, :], reference.body_quat[..., a, :]) / 0.4**2),
        "action_rate": -0.1 * jnp.sum((action - previous_action) ** 2, axis=-1),
        "joint_limit": -2.0 * jnp.sum(jnp.maximum(lower - robot.joint_pos, 0) + jnp.maximum(robot.joint_pos - upper, 0), axis=-1),
    }
    terms = {name: value * dt for name, value in terms.items()}
    return sum(terms.values()), terms
