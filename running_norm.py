"""Pure JAX running moments, with detached finite samples and count limits."""

from typing import NamedTuple
import jax
import jax.numpy as jnp


class RunningMoments(NamedTuple):
    mean: object
    variance: object
    count: object


def empty_moments(size):
    return RunningMoments(jnp.zeros(size), jnp.ones(size), jnp.array(0., jnp.float32))


def update_moments(stats, values, mask=None, extra_variance=0., max_count=2e8):
    values = jax.lax.stop_gradient(jnp.asarray(values, jnp.float32))
    valid = jnp.isfinite(values).all(-1)
    if mask is not None:
        valid &= mask
    valid &= stats.count < max_count
    axes = tuple(range(values.ndim - 1))
    count = valid.sum(dtype=jnp.float32)
    mean = jnp.where(valid[..., None], values, 0.).sum(axes) / jnp.maximum(count, 1.)
    centered = jnp.where(valid[..., None], values - mean, 0.)
    variance = (centered ** 2).sum(axes) / jnp.maximum(count, 1.) + extra_variance
    total = jnp.maximum(stats.count + count, 1.)
    delta = mean - stats.mean
    merged_mean = stats.mean + delta * count / total
    merged_var = (stats.variance * stats.count + variance * count
                  + delta ** 2 * stats.count * count / total) / total
    return RunningMoments(jnp.where(count > 0, merged_mean, stats.mean),
                           jnp.where(count > 0, merged_var, stats.variance), stats.count + count)


def normalize(values, stats, eps=.01):
    return (values - stats.mean) / jnp.sqrt(stats.variance + eps**2)
