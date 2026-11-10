"""State-inclusive controlled Koopman model with fixed coordinate projection."""

from dataclasses import dataclass
from math import isfinite

from flax import linen as nn
import jax
import jax.numpy as jnp

from networks import MLP


@dataclass(frozen=True)
class KoopmanConfig:
    state_dim: int = 71
    action_dim: int = 29
    context_dim: int = 33
    feature_dim: int = 128
    encoder_hidden: tuple[int, ...] = (256, 256)
    matrix_noise: float = 1e-3
    one_step_weight: float = 1.
    multi_step_weight: float = 1.
    latent_weight: float = .1
    dt: float = .02
    local_frame = True
    architecture = "koopman"

    def __post_init__(self):
        object.__setattr__(self, 'encoder_hidden', tuple(self.encoder_hidden))
        if any(not isinstance(n, int) or n < 1 for n in
               (self.state_dim, self.action_dim, self.feature_dim) + self.encoder_hidden):
            raise ValueError('State/action/feature dimensions and encoder widths must be positive integers')
        if not isinstance(self.context_dim, int) or self.context_dim < 0:
            raise ValueError('context_dim must be a nonnegative integer')
        weights = (self.one_step_weight, self.multi_step_weight, self.latent_weight)
        if any(not isfinite(w) or w < 0 for w in weights) or sum(weights[:2]) <= 0:
            raise ValueError('Loss weights must be finite/nonnegative, with a state prediction loss enabled')
        if not isfinite(self.matrix_noise) or self.matrix_noise <= 0 or not isfinite(self.dt) or self.dt <= 0:
            raise ValueError('matrix_noise and dt must be positive and finite')

    @property
    def latent_dim(self):
        return self.state_dim + self.feature_dim


class Koopman(nn.Module):
    config: KoopmanConfig = KoopmanConfig()

    def setup(self):
        c = self.config
        self.encoder = MLP(c.encoder_hidden, c.feature_dim, 'silu')
        def near_identity(key, shape, dtype=jnp.float32):
            return jnp.eye(shape[0], dtype=dtype) + c.matrix_noise * jax.random.normal(key, shape, dtype)
        self.A = self.param('A', near_identity, (c.latent_dim, c.latent_dim))
        self.B = self.param('B', nn.initializers.normal(c.matrix_noise), (c.latent_dim, c.action_dim))
        self.C = self.param('C', nn.initializers.normal(c.matrix_noise), (c.latent_dim, c.context_dim))

    def encode(self, state):
        if state.shape[-1] != self.config.state_dim:
            raise ValueError('State dimension does not match Koopman config')
        return jnp.concatenate((state, self.encoder(state)), -1)

    def advance(self, latent, action, context):
        c = self.config
        if latent.shape[-1] != c.latent_dim or action.shape != latent.shape[:-1] + (c.action_dim,) or context.shape != latent.shape[:-1] + (c.context_dim,):
            raise ValueError('Latent/action/context shapes do not match Koopman config')
        return latent @ self.A.T + action @ self.B.T + context @ self.C.T

    def decode(self, latent):
        # P=[I,0]: no decoder parameters and no nonlinear projection in latent space.
        return latent[..., :self.config.state_dim]

    def __call__(self, state, action, context):
        return self.decode(self.advance(self.encode(state), action, context))

    def rollout(self, initial, actions, context):
        """Batch-first actions/contexts (B,T,*); encode only the initial state."""
        if actions.ndim != 3 or context.shape != actions.shape[:2] + (self.config.context_dim,):
            raise ValueError('Expected actions and contexts with batch/time axes')
        def step(z, inputs):
            z = self.advance(z, *inputs)
            return z, z
        _, latent = jax.lax.scan(step, self.encode(initial),
                                 (jnp.swapaxes(actions, 0, 1), jnp.swapaxes(context, 0, 1)))
        latent = jnp.swapaxes(latent, 0, 1)
        return self.decode(latent), latent
