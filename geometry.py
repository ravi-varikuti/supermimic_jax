"""Batched rotation operations and G1 local-frame state representation."""

import jax.numpy as jnp
from normalization import unit_quaternion


def conjugate(q):
    return q * jnp.array([1., -1., -1., -1.])


def multiply(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return jnp.concatenate((aw * bw - (av * bv).sum(-1, keepdims=True),
                            aw * bv + bw * av + jnp.cross(av, bv)), -1)


def rotate(q, vector):
    t = 2. * jnp.cross(q[..., 1:], vector)
    return vector + q[..., :1] * t + jnp.cross(q[..., 1:], t)


def heading(q):
    w, x, y, z = (q[..., i] for i in range(4))
    yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y*y + z*z))
    zero = jnp.zeros_like(yaw)
    return jnp.stack((jnp.cos(yaw / 2), zero, zero, jnp.sin(yaw / 2)), -1)


def rotation6d(q):
    return jnp.concatenate((rotate(q, jnp.array([1., 0., 0.])),
                            rotate(q, jnp.array([0., 1., 0.]))), -1)


def exp_rotation(vector):
    angle = jnp.sqrt(jnp.maximum((vector * vector).sum(-1, keepdims=True), 1e-12))
    return unit_quaternion(jnp.concatenate((jnp.cos(angle / 2),
        .5 * jnp.sinc(angle / (2 * jnp.pi)) * vector), -1))


def log_rotation(q):
    q = unit_quaternion(q)
    q = q * jnp.where(q[..., :1] < 0, -1., 1.)
    squared = (q[..., 1:] ** 2).sum(-1, keepdims=True)
    length = jnp.sqrt(jnp.maximum(squared, 1e-12))
    scale = 2 * jnp.arctan2(length, q[..., :1]) / length
    return jnp.where(squared < 1e-8, 2 * q[..., 1:], scale * q[..., 1:])


def invariant_state(state):
    """71 features: height, heading-free rotation, local velocities, joints."""
    q = unit_quaternion(state[..., 3:7])
    return jnp.concatenate((state[..., 2:3], rotation6d(multiply(conjugate(heading(q)), q)),
                            rotate(conjugate(q), state[..., 36:39]), state[..., 39:42],
                            state[..., 7:36], state[..., 42:]), -1)


def local_delta(state, following):
    """70 physical increments; free-joint angular velocity is stored body-local."""
    q, q1 = unit_quaternion(state[..., 3:7]), unit_quaternion(following[..., 3:7])
    w, w1 = rotate(q, state[..., 39:42]), rotate(q1, following[..., 39:42])
    return jnp.concatenate((rotate(conjugate(heading(q)), following[..., :3] - state[..., :3]),
        log_rotation(multiply(conjugate(q), q1)),
        rotate(conjugate(q), following[..., 36:39] - state[..., 36:39]),
        rotate(conjugate(q), w1 - w), following[..., 7:36] - state[..., 7:36],
        following[..., 42:] - state[..., 42:]), -1)


def integrate_local(state, delta):
    q = unit_quaternion(state[..., 3:7])
    q1 = unit_quaternion(multiply(q, exp_rotation(delta[..., 3:6])))
    velocity = state[..., 36:39] + rotate(q, delta[..., 6:9])
    omega_world = rotate(q, state[..., 39:42]) + rotate(q, delta[..., 9:12])
    return jnp.concatenate((state[..., :3] + rotate(heading(q), delta[..., :3]), q1,
                            state[..., 7:36] + delta[..., 12:41], velocity,
                            rotate(conjugate(q1), omega_world), state[..., 42:] + delta[..., 41:]), -1)


def normalize_context(context):
    mean = jnp.array([.75] + [0.] * 32)
    scale = jnp.array([.45] + [.01] * 29 + [.05] * 3)
    return (context - mean) / scale
