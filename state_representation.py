"""Pack configured fields into JAX vectors; preserve leading batch/time axes."""

from math import prod

import jax.numpy as jnp


def _pack(values, shapes):
    parts, leading = [], None
    for name, shape in shapes:
        value = jnp.asarray(values[name], dtype=jnp.float32)
        if shape and (value.ndim < len(shape) or value.shape[-len(shape):] != shape):
            raise ValueError(f"{name} must end with shape {shape}; got {value.shape}")
        batch = value.shape[:-len(shape)] if shape else value.shape
        if leading is not None and batch != leading:
            raise ValueError(f"{name} has leading shape {batch}; expected {leading}")
        leading = batch
        parts.append(value.reshape(batch + (prod(shape),)))
    return jnp.concatenate(parts, axis=-1)


def _unpack(vector, shapes):
    vector = jnp.asarray(vector, dtype=jnp.float32)
    size = sum(prod(shape) for _, shape in shapes)
    if vector.ndim < 1 or vector.shape[-1] != size:
        raise ValueError(f"Expected final dimension {size}; got {vector.shape}")
    fields, start = {}, 0
    for name, shape in shapes:
        end = start + prod(shape)
        fields[name] = vector[..., start:end].reshape(vector.shape[:-1] + shape)
        start = end
    return fields


def pack_state(snapshot, config):
    """Pack physical state only; actions, references, and history remain separate."""
    return _pack(snapshot, config.state_shapes)


def unpack_state(vector, config):
    return _unpack(vector, config.state_shapes)


def pack_context(context, config, *, batch_shape=()):
    """For robots without randomization, supply batch_shape to get (..., 0)."""
    if not config.context_shapes:
        return jnp.zeros(tuple(batch_shape) + (0,), dtype=jnp.float32)
    return _pack(context, config.context_shapes)


def unpack_context(vector, config):
    return _unpack(vector, config.context_shapes)
