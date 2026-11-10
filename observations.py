"""Proprioceptive observations; functions support batched JAX arrays."""

import jax
import jax.numpy as jnp


def projected_gravity(qpos):
    """World gravity direction expressed in the pelvis frame (wxyz quaternion)."""
    w, x, y, z = (qpos[..., i] for i in range(3, 7))
    return jnp.stack((2 * (w * y - x * z),
                      -2 * (y * z + w * x),
                      2 * (x * x + y * y) - 1), axis=-1)


def proprioception(qpos, qvel, previous_action, default_joint_pos, joint_bias=0.0, reference_qpos=None):
    """93 proprioceptive values, plus six relative-rotation values with a reference.

    default_joint_pos is the controller's nominal pose, not the reference pose.
    joint_bias is the fixed encoder calibration error sampled for the episode.
    Free-joint angular velocity qvel[3:6] is in the pelvis frame, matching its IMU.
    """
    base = jnp.concatenate((
        qpos[..., 7:] - joint_bias - default_joint_pos,  # 29
        qvel[..., 6:],                    # 29
        qvel[..., 3:6],                   # 3
        projected_gravity(qpos),          # 3
        previous_action,                 # 29
    ), axis=-1)
    if reference_qpos is None:
        return base
    from geometry import multiply, conjugate, rotation6d
    from normalization import unit_quaternion
    relative = multiply(conjugate(unit_quaternion(qpos[..., 3:7])),
                        unit_quaternion(reference_qpos[..., 3:7]))
    return jnp.concatenate((base, rotation6d(relative)), -1)


def noise_scale(reference_orientation=False):
    """G1 sensor noise amplitudes in the same order as proprioception."""
    return jnp.array([0.01] * 29 + [0.5] * 29 + [0.2] * 3 + [0.05] * 3 + [0.0] * 29
                     + ([.05] * 6 if reference_orientation else []))


def add_noise(observation, key):
    """SuperMimic's uniform sensor noise; previous actions stay noise-free."""
    return observation + jax.random.uniform(key, observation.shape, minval=-1, maxval=1) * noise_scale(observation.shape[-1] == 99)


def initialize_history(observation, length=10):
    """Repeat the initial observation: (..., length, features), oldest first."""
    return jnp.repeat(observation[..., None, :], length, axis=-2)


def update_history(history, observation):
    """Discard the oldest frame and append the current observation."""
    return jnp.concatenate((history[..., 1:, :], observation[..., None, :]), axis=-2)
