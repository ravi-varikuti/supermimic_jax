"""Deterministic goal-conditioned policy; robot dimensions come from config."""

from dataclasses import dataclass

from flax import linen as nn
import jax.numpy as jnp

from networks import MLP
from robot_config import RobotConfig
from running_norm import normalize


@dataclass(frozen=True)
class PolicyConfig:
    future_frames: int = 10
    goal_frame_dim: int | None = None  # Default: packed physical state dimension.
    latent_dim: int = 32
    encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    activation: str = "elu"
    bound_actions: bool = True

    def __post_init__(self):
        dimensions = (self.future_frames, self.latent_dim) + self.encoder_hidden + self.decoder_hidden
        if self.goal_frame_dim is not None:
            dimensions += (self.goal_frame_dim,)
        if any(not isinstance(size, int) or size < 1 for size in dimensions):
            raise ValueError("Policy dimensions must be positive integers")
        if self.activation not in ("elu", "relu", "silu", "tanh"):
            raise ValueError(f"Unknown activation: {self.activation}")


class MotionPolicy(nn.Module):
    robot_config: RobotConfig
    config: PolicyConfig = PolicyConfig()

    @nn.compact
    def __call__(self, observation, goal, normalization=None):
        observation, goal = jnp.asarray(observation, jnp.float32), jnp.asarray(goal, jnp.float32)
        if observation.ndim < 1 or observation.shape[-1] != self.robot_config.policy_observation_dim:
            raise ValueError(f"observation must end with {self.robot_config.policy_observation_dim} values")
        batch = observation.shape[:-1]
        goal_dim = self.config.goal_frame_dim or self.robot_config.state_dim
        if goal.shape != batch + (self.config.future_frames, goal_dim):
            raise ValueError(f"goal must have shape {batch + (self.config.future_frames, goal_dim)}")
        if normalization is not None:
            observation = normalize(observation.reshape(batch + (self.robot_config.history_length,
                                                                  self.robot_config.observation_dim)), normalization.observation).reshape(observation.shape)
            goal = normalize(goal, normalization.goal)
        encoded = MLP(self.config.encoder_hidden, self.config.latent_dim,
                      self.config.activation, name="motion_encoder")(
                          goal.reshape(batch + (self.config.future_frames * goal_dim,)))
        latent = jnp.tanh(encoded)
        inputs = jnp.concatenate((observation, latent), axis=-1)
        # Small initial output keeps the untrained policy near zero joint offsets.
        action = MLP(self.config.decoder_hidden, self.robot_config.action_dim,
                     self.config.activation, output_kernel_scale=.01, name="decoder")(inputs)
        return jnp.tanh(action) if self.config.bound_actions else action
