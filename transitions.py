"""Robot-independent replay records built before any episode reset."""

from typing import NamedTuple

import jax.numpy as jnp

from state_representation import pack_context, pack_state


class Transition(NamedTuple):
    state: object
    action: object
    next_state: object
    reward: object
    terminated: object
    truncated: object
    context: object
    previous_action: object
    observation: object
    next_observation: object
    metadata: dict

    @property
    def done(self):
        return self.terminated | self.truncated

    @property
    def bootstrap_mask(self):
        return ~self.terminated


def _array(value, shape, dtype, name):
    value = jnp.asarray(value, dtype=dtype)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {value.shape}")
    return value


def make_transition(before, after, reward, terminated, truncated, config, *, metadata=None):
    """Pack two snapshots of the same episode, with `after` taken before reset.

    Actions come from after['previous_action']: the command actually executed,
    including clipping and any future collection noise. Observations are the
    newest clean frames; full history will be reconstructed when sampling replay.
    Leading axes may be environment, time/environment, or absent.
    """
    state = pack_state(before, config)
    batch = state.shape[:-1]
    vector = lambda value, size, name: _array(value, batch + (size,), jnp.float32, name)
    history_shape = batch + (config.history_length, config.observation_dim)
    history = _array(before["clean_observation_history"], history_shape, jnp.float32, "clean history")
    next_history = _array(after["clean_observation_history"], history_shape, jnp.float32, "next clean history")
    context = pack_context(before["domain_randomization"], config, batch_shape=batch)
    return Transition(
        state=state,
        action=vector(after["previous_action"], config.action_dim, "executed action"),
        next_state=vector(pack_state(after, config), config.state_dim, "next state"),
        reward=_array(reward, batch, jnp.float32, "reward"),
        terminated=_array(terminated, batch, bool, "terminated"),
        truncated=_array(truncated, batch, bool, "truncated"),
        context=vector(context, config.context_dim, "context"),
        previous_action=vector(before["previous_action"], config.action_dim, "previous action"),
        observation=history[..., -1, :],
        next_observation=next_history[..., -1, :],
        metadata={} if metadata is None else dict(metadata),
    )
