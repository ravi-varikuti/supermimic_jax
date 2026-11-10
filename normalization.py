"""Fixed warm-up statistics and configurable quaternion handling."""

from typing import NamedTuple

import jax.numpy as jnp


class FeatureStats(NamedTuple):
    mean: object
    scale: object


class Normalization(NamedTuple):
    state: FeatureStats
    context: FeatureStats
    delta: FeatureStats
    count: object


def unit_quaternion(quaternion):
    squared = jnp.sum(quaternion ** 2, axis=-1, keepdims=True)
    normalized = quaternion / jnp.sqrt(jnp.maximum(squared, 1e-12))
    identity = jnp.array([1., 0., 0., 0.], dtype=quaternion.dtype)
    return jnp.where(squared > 1e-12, normalized, identity)


def prepare_state(state, config, *, canonical=True):
    """Normalize configured wxyz quaternions; optionally choose nonnegative w."""
    for start, end in config.quaternion_slices:
        quaternion = unit_quaternion(state[..., start:end])
        if canonical:
            quaternion *= jnp.where(quaternion[..., :1] < 0., -1., 1.)
        state = state.at[..., start:end].set(quaternion)
    return state


def align_state(target, reference, config):
    """Choose equivalent quaternion signs nearest the reference orientation."""
    target = prepare_state(target, config, canonical=False)
    for start, end in config.quaternion_slices:
        quaternion = target[..., start:end]
        dot = jnp.sum(quaternion * reference[..., start:end], axis=-1, keepdims=True)
        target = target.at[..., start:end].set(quaternion * jnp.where(dot < 0., -1., 1.))
    return target


def _statistics(values, mask, min_std):
    axes = tuple(range(values.ndim - 1))
    count = jnp.maximum(mask.sum(), 1)
    mean = jnp.where(mask[..., None], values, 0.).sum(axis=axes) / count
    centered = jnp.where(mask[..., None], values - mean, 0.)
    variance = (centered ** 2).sum(axis=axes) / count
    return FeatureStats(mean, jnp.maximum(jnp.sqrt(variance), min_std))


def _exclude_quaternions(stats, config):
    mean, scale = stats
    for start, end in config.quaternion_slices:
        mean, scale = mean.at[start:end].set(0.), scale.at[start:end].set(1.)
    return FeatureStats(mean, scale)


def fit_normalization(transitions, config, mask=None, min_std=1e-3):
    """Fit once from warm-up records. Ignore masked/nonfinite transitions.

    State statistics use current and next states; delta statistics use aligned
    next_state - state. Quaternion coordinates always keep mean=0 and scale=1.
    """
    if min_std <= 0:
        raise ValueError("min_std must be positive")
    mask = jnp.ones(transitions.reward.shape, bool) if mask is None else jnp.asarray(mask, bool)
    if mask.shape != transitions.reward.shape:
        raise ValueError("mask must have the same shape as rewards")
    for values in (transitions.state, transitions.next_state, transitions.action, transitions.context):
        mask = mask & jnp.isfinite(values).all(axis=-1)
    current = prepare_state(transitions.state, config)
    following = align_state(transitions.next_state, current, config)
    paired = jnp.stack((current, following), axis=-2)
    paired_mask = jnp.broadcast_to(mask[..., None], paired.shape[:-1])
    return Normalization(
        _exclude_quaternions(_statistics(paired, paired_mask, min_std), config),
        _statistics(transitions.context, mask, min_std),
        _exclude_quaternions(_statistics(following - current, mask, min_std), config),
        mask.sum(),
    )


def fit_replay_normalization(storage, config, min_std=1e-3):
    """Exclude unwritten ring slots; buffers that have wrapped use every row."""
    capacity = storage.data.reward.shape[0]
    oldest = (storage.write_index - storage.size) % capacity
    filled = (jnp.arange(capacity) - oldest) % capacity < storage.size
    mask = jnp.broadcast_to(filled[:, None], storage.data.reward.shape)
    return fit_normalization(storage.data, config, mask, min_std)
