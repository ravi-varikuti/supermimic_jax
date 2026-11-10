"""Privileged value network; deployment uses only the policy."""

from flax import linen as nn
import jax.numpy as jnp

from networks import MLP
from robot_config import RobotConfig
from geometry import invariant_state, normalize_context, rotate, heading, conjugate, multiply, rotation6d
from running_norm import normalize


class ValueCritic(nn.Module):
    robot_config: RobotConfig
    goal_frames: int = 10
    goal_frame_dim: int | None = None
    hidden_dims: tuple[int, ...] = (256, 128)
    local_frame: bool = False

    def features(self, state, goal, context, reference_state=None):
        batch = state.shape[:-1]
        if not self.local_frame:
            return jnp.concatenate((state, goal.reshape(batch + (-1,)), context), -1)
        if reference_state is None:
            raise ValueError("Local critic requires the current reference state")
        h_inv = conjugate(heading(state[..., 3:7]))
        errors = jnp.concatenate((rotate(h_inv, reference_state[..., :3] - state[..., :3]),
            rotation6d(multiply(conjugate(state[..., 3:7]), reference_state[..., 3:7])),
            reference_state[..., 7:36] - state[..., 7:36]), -1)
        return jnp.concatenate((invariant_state(state), goal.reshape(batch + (-1,)),
                                normalize_context(context), errors), -1)

    @nn.compact
    def __call__(self, state, goal, context, normalization=None, reference_state=None):
        batch = state.shape[:-1]
        dim = self.goal_frame_dim or self.robot_config.state_dim
        if state.shape[-1] != self.robot_config.state_dim:
            raise ValueError("Critic state shape does not match robot config")
        if goal.shape != batch + (self.goal_frames, dim):
            raise ValueError("Critic goal shape does not match config")
        if context.shape != batch + (self.robot_config.context_dim,):
            raise ValueError("Critic context shape does not match robot config")
        inputs = self.features(state, goal, context, reference_state)
        if normalization is not None:
            inputs = normalize(inputs, normalization)
        return MLP(self.hidden_dims, 1, name="value")(inputs)[..., 0]
