"""Configurable dynamics MLP: physical state, action, context -> next state."""

from dataclasses import dataclass

from flax import linen as nn
import jax.numpy as jnp

from networks import MLP
from normalization import prepare_state
from robot_config import RobotConfig


def build_world_model(robot, config, *, local_frame=True, dt=.02):
    """Select an architecture using the saved RunConfig; MLP remains the default."""
    if config.world_model_type == 'koopman':
        from koopman import KoopmanConfig
        from koopman_g1 import KoopmanWorldModel
        return KoopmanWorldModel(robot, KoopmanConfig(
            feature_dim=config.koopman_features, encoder_hidden=config.koopman_encoder_hidden,
            one_step_weight=config.koopman_one_step_weight, multi_step_weight=config.koopman_multi_step_weight,
            latent_weight=config.koopman_latent_weight, dt=dt))
    return WorldModel(robot, WorldModelConfig(hidden_dims=config.dynamics_hidden, local_frame=local_frame))


@dataclass(frozen=True)
class WorldModelConfig:
    hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
    activation: str = "elu"
    predict_delta: bool = True
    local_frame: bool = False

    def __post_init__(self):
        if any(not isinstance(width, int) or width < 1 for width in self.hidden_dims):
            raise ValueError("Hidden layer widths must be positive integers")
        if self.activation not in ("elu", "relu", "silu", "tanh"):
            raise ValueError(f"Unknown activation: {self.activation}")


class WorldModel(nn.Module):
    robot_config: RobotConfig
    config: WorldModelConfig = WorldModelConfig()

    @nn.compact
    def __call__(self, state, action, context, normalization=None):
        state, action, context = (jnp.asarray(value, jnp.float32)
                                  for value in (state, action, context))
        if state.ndim < 1 or state.shape[-1] != self.robot_config.state_dim:
            raise ValueError(f"state must end with dimension {self.robot_config.state_dim}")
        batch = state.shape[:-1]
        if action.shape != batch + (self.robot_config.action_dim,):
            raise ValueError(f"action must have shape {batch + (self.robot_config.action_dim,)}")
        if context.shape != batch + (self.robot_config.context_dim,):
            raise ValueError(f"context must have shape {batch + (self.robot_config.context_dim,)}")
        state = prepare_state(state, self.robot_config)
        if self.config.local_frame:
            from geometry import invariant_state, normalize_context, integrate_local
            if state.shape[-1] != 71:
                raise ValueError("Local qpos/qvel adapter currently expects a 29-joint floating-base robot")
            encoded = invariant_state(state)
            if normalization is not None:
                encoded = (encoded - normalization.state.mean) / normalization.state.scale
            inputs = jnp.concatenate((encoded, action, normalize_context(context)), -1)
            delta = MLP(self.config.hidden_dims, 70, "silu", layer_norm=True,
                        zero_output=True, name="dynamics")(inputs)
            if normalization is not None:
                delta = delta * normalization.delta.scale + normalization.delta.mean
            return integrate_local(state, delta)
        encoded_state = state
        if normalization is not None:
            encoded_state = (state - normalization.state.mean) / normalization.state.scale
            context = (context - normalization.context.mean) / normalization.context.scale
        inputs = jnp.concatenate((encoded_state, action, context), axis=-1)
        prediction = MLP(
            hidden_dims=self.config.hidden_dims,
            output_dim=self.robot_config.state_dim,
            activation=self.config.activation,
            name="dynamics",
        )(inputs)
        if normalization is not None:
            stats = normalization.delta if self.config.predict_delta else normalization.state
            prediction = prediction * stats.scale + stats.mean
        following = state + prediction if self.config.predict_delta else prediction
        return prepare_state(following, self.robot_config, canonical=False)
