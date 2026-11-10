"""JAX circular replay, with time rows containing all parallel environments."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from transitions import Transition


class ReplayState(NamedTuple):
    data: Transition
    episode_steps: object
    next_episode_steps: object
    write_index: object
    size: object


class ReplaySample(NamedTuple):
    transitions: Transition
    history: object
    valid: object
    num_valid: object


class ReplayBuffer:
    def __init__(self, config, capacity=1000, sequence_length=8):
        """capacity is control steps per environment, not total transitions."""
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if not isinstance(sequence_length, int) or not 1 <= sequence_length <= capacity:
            raise ValueError("sequence_length must be between 1 and capacity")
        self.config = config
        self.capacity = capacity
        self.sequence_length = sequence_length

    def init(self, example):
        """Allocate from one Transition shaped (num_envs, ...), including metadata."""
        if example.reward.ndim != 1 or example.reward.shape[0] < 1:
            raise ValueError("example must contain one batched environment step")
        count = example.reward.shape[0]
        if example.terminated.shape != (count,) or example.truncated.shape != (count,):
            raise ValueError("Ending flags must have shape (num_envs,)")

        def allocate(value):
            value = jnp.asarray(value)
            if not value.shape or value.shape[0] != count:
                raise ValueError("All transition fields must have a leading environment axis")
            return jnp.zeros((self.capacity,) + value.shape, dtype=value.dtype)

        for name, size in (("state", self.config.state_dim), ("next_state", self.config.state_dim),
                           ("action", self.config.action_dim), ("previous_action", self.config.action_dim),
                           ("context", self.config.context_dim), ("observation", self.config.observation_dim),
                           ("next_observation", self.config.observation_dim)):
            if getattr(example, name).shape != (count, size):
                raise ValueError(f"{name} must have shape {(count, size)}")
        zero = jnp.array(0, jnp.int32)
        return ReplayState(
            jax.tree.map(allocate, example), jnp.zeros((self.capacity, count), jnp.int32),
            jnp.zeros(count, jnp.int32), zero, zero,
        )

    def add(self, state, transition):
        """Write one time row. Supply records in collection order, before reset."""
        def write(stored, value):
            if value.shape != stored.shape[1:]:
                raise ValueError(f"Expected transition shape {stored.shape[1:]}; got {value.shape}")
            return stored.at[state.write_index].set(value)

        # Optional metadata supports starting collection mid-episode and detects
        # manual resets/gaps. Otherwise derive counters from the ending flags.
        steps = transition.metadata.get("episode_steps", state.next_episode_steps)
        steps = jnp.asarray(steps, jnp.int32)
        if steps.shape != state.next_episode_steps.shape:
            raise ValueError("episode_steps metadata must have shape (num_envs,)")
        return ReplayState(
            jax.tree.map(write, state.data, transition),
            state.episode_steps.at[state.write_index].set(steps),
            jnp.where(transition.done, 0, steps + 1),
            (state.write_index + 1) % self.capacity,
            jnp.minimum(state.size + 1, self.capacity),
        )

    def extend(self, state, transitions):
        """Append a collection shaped (time, num_envs, ...) using scan."""
        return jax.lax.scan(lambda carry, item: (self.add(carry, item), None),
                            state, transitions)[0]

    def valid_starts(self, state):
        """Return a chronological (capacity, num_envs) eligibility mask."""
        time = jnp.arange(self.capacity)
        oldest = (state.write_index - state.size) % self.capacity
        slots = (oldest + time) % self.capacity
        steps = state.episode_steps[slots]
        breaks = state.data.done[slots]
        # Discontinuous episode counters also break a stream, even if the caller
        # manually reset without a preceding termination/truncation flag.
        breaks = breaks.at[:-1].set(breaks[:-1] | (steps[1:] != steps[:-1] + 1))
        prefix = jnp.concatenate((jnp.zeros_like(steps[:1]),
                                  jnp.cumsum(breaks, axis=0, dtype=jnp.int32)), axis=0)
        end = jnp.minimum(time + self.sequence_length - 1, self.capacity)
        sequence_ok = prefix[end] == prefix[time]  # Final transition may be terminal.
        lookback = jnp.minimum(steps, self.config.history_length - 1)
        history_start = time[:, None] - lookback
        worlds = jnp.arange(steps.shape[1])[None, :]
        history_ok = (history_start >= 0) & (
            prefix[time] == prefix[jnp.maximum(history_start, 0), worlds]
        )
        filled = time + self.sequence_length <= state.size
        return filled[:, None] & sequence_ok & history_ok

    def sample(self, state, key, batch_size):
        """Sample with replacement. Check result.valid before using training data.

        Returns sequences (batch, sequence_length, ...) and initial clean history
        (batch, history_length, observation_dim). No eligible starts yields zeros
        with valid=False. batch_size must be static under jit.
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        eligible = self.valid_starts(state)
        cumulative = jnp.cumsum(eligible.reshape(-1), dtype=jnp.int32)
        count = cumulative[-1]
        draws = jax.random.randint(key, (batch_size,), 0, jnp.maximum(count, 1))
        # Inverse CDF sampling avoids a batch_size x capacity x num_envs allocation.
        flat = jnp.searchsorted(cumulative, draws, side="right")
        flat = jnp.minimum(flat, cumulative.size - 1)
        worlds = flat % eligible.shape[1]
        time = flat // eligible.shape[1]
        oldest = (state.write_index - state.size) % self.capacity
        starts = (oldest + time) % self.capacity
        slots = (starts[:, None] + jnp.arange(self.sequence_length)) % self.capacity
        records = jax.tree.map(lambda value: value[slots, worlds[:, None]], state.data)
        steps = state.episode_steps[starts, worlds]
        lookback = jnp.arange(self.config.history_length - 1, -1, -1)
        history_slots = (starts[:, None] - jnp.minimum(lookback, steps[:, None])) % self.capacity
        history = state.data.observation[history_slots, worlds[:, None]]
        ready = count > 0
        records = jax.tree.map(lambda value: jnp.where(ready, value, jnp.zeros_like(value)), records)
        history = jnp.where(ready, history, jnp.zeros_like(history))
        return ReplaySample(records, history, ready, count)
