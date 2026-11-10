"""Analytical observations and clean/noisy histories for imagined rollouts."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from observations import update_history


class ObservationHistory(NamedTuple):
    clean: object
    noisy: object
    key: object

    @property
    def observation(self):
        """Flattened noisy history passed to the policy."""
        return self.noisy.reshape(self.noisy.shape[:-2] + (self.noisy.shape[-2] * self.noisy.shape[-1],))


class ObservationModel:
    def __init__(self, config, observation_fn, noise_scale=None, noise=True):
        """observation_fn takes raw packed state, executed action, and context."""
        self.config, self.observation_fn, self.noise = config, observation_fn, noise
        self.noise_scale = jnp.zeros(config.observation_dim, jnp.float32) if noise_scale is None else jnp.asarray(noise_scale, jnp.float32)
        if self.noise_scale.shape != (config.observation_dim,):
            raise ValueError(f"noise_scale must have shape ({config.observation_dim},)")

    def clean(self, state, previous_action, context, reference_state=None):
        """Reconstruct one frame; previous_action produced the supplied state."""
        state = jnp.asarray(state, jnp.float32)
        previous_action, context = jnp.asarray(previous_action, jnp.float32), jnp.asarray(context, jnp.float32)
        if state.ndim < 1 or state.shape[-1] != self.config.state_dim:
            raise ValueError(f"state must end with {self.config.state_dim} values")
        batch = state.shape[:-1]
        if previous_action.shape != batch + (self.config.action_dim,):
            raise ValueError("previous_action shape does not match state/config")
        if context.shape != batch + (self.config.context_dim,):
            raise ValueError("context shape does not match state/config")
        args = (state, previous_action, context)
        if reference_state is not None:
            args += (reference_state,)
        clean = jnp.asarray(self.observation_fn(*args), jnp.float32)
        if clean.shape != batch + (self.config.observation_dim,):
            raise ValueError("Reconstructed observation shape does not match config")
        return clean

    def init_history(self, clean_history, key, episode_steps=None):
        """Seed from ReplaySample.history and resample sensor noise.

        Supply episode_steps at the start of each sample to share the initial
        noise across repeated reset-padding frames, matching environment resets.
        """
        clean_history = jnp.asarray(clean_history, jnp.float32)
        shape = (self.config.history_length, self.config.observation_dim)
        if clean_history.ndim < 2 or clean_history.shape[-2:] != shape:
            raise ValueError(f"clean_history must end with shape {shape}")
        key, noise_key = jax.random.split(key)
        noise = jax.random.uniform(noise_key, clean_history.shape, dtype=jnp.float32,
                                   minval=-1., maxval=1.) * self.noise_scale
        if episode_steps is not None:
            steps = jnp.asarray(episode_steps, jnp.int32)
            if steps.shape != clean_history.shape[:-2]:
                raise ValueError("episode_steps must match history batch axes")
            slots = jnp.arange(self.config.history_length)
            padding = self.config.history_length - 1 - steps[..., None]
            source = jnp.where(slots <= padding, 0, slots)
            indices = jnp.broadcast_to(source[..., None], clean_history.shape)
            noise = jnp.take_along_axis(noise, indices, axis=-2)
        noisy = clean_history + noise if self.noise else clean_history
        return ObservationHistory(clean_history, noisy, key)

    def step(self, history, next_state, action, context, reference_state=None):
        """Append a predicted state's observation; keep older sensor noise intact."""
        clean = self.clean(next_state, action, context, reference_state)
        key, noise_key = jax.random.split(history.key)
        noise = jax.random.uniform(noise_key, clean.shape, dtype=jnp.float32,
                                   minval=-1., maxval=1.) * self.noise_scale
        noisy = clean + noise if self.noise else clean
        return ObservationHistory(update_history(history.clean, clean),
                                  update_history(history.noisy, noisy), key)
