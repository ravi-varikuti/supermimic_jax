"""Pure policy rollouts with exploration, terminal records, and masked resets."""

from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp

from transitions import make_transition


class CollectorState(NamedTuple):
    env_state: object
    key: object  # Exploration RNG is separate from the environment's sensor/reset RNG.


class PolicyCollector:
    def __init__(self, env, policy, goal_fn, metadata_fn=None, exploration_std=.1, reset_sampler=None):
        if not isfinite(exploration_std) or exploration_std < 0:
            raise ValueError("exploration_std must be finite and nonnegative")
        if env.robot_config != policy.robot_config:
            raise ValueError("Policy and environment robot configurations must match")
        self.env, self.policy = env, policy
        self.goal_fn, self.metadata_fn = goal_fn, metadata_fn
        self.exploration_std = exploration_std
        self.reset_sampler = reset_sampler

    def collect(self, variables, carry, num_steps, reset_weights=None, normalization=None):
        """Return (carry, records), with records shaped (time, environment, ...).

        num_steps is static under jit. This function does not mutate env.state.
        Goal/metadata callbacks keep task-specific reference details outside it.
        """
        if not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")

        def advance(carry, _):
            state = carry.env_state
            key, noise_key = jax.random.split(carry.key)
            observation = state.observation_history.reshape(self.env.num_envs, -1)
            args = (variables, observation, self.goal_fn(state))
            action = self.policy.apply(*args) if normalization is None else self.policy.apply(*args, normalization)
            noise = self.exploration_std * jax.random.normal(noise_key, action.shape)
            limit = getattr(self.env, "action_limit", 1.)
            executed = jnp.clip(action + noise, -limit, limit)
            terminal, _, reward, terminated, truncated, _ = self.env.step_jax(state, executed)
            before, after = self.env.get_state(state), self.env.get_state(terminal)
            metadata = None if self.metadata_fn is None else self.metadata_fn(before, after)
            record = make_transition(before, after, reward, terminated, truncated,
                                     self.env.robot_config, metadata=metadata)
            if self.reset_sampler is None:
                following, _, _ = self.env.reset_jax(terminal, terminated | truncated)
            else:
                key, reset_key = jax.random.split(key)
                ids, frames = self.reset_sampler(reset_key, reset_weights, self.env.num_envs)
                following, _, _ = self.env.reset_jax(terminal, terminated | truncated, ids, frames)
            return CollectorState(following, key), record

        return jax.lax.scan(advance, carry, None, length=num_steps)
