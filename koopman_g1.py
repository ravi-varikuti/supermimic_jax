"""Physical G1 adapter for linear Koopman dynamics in normalized local features."""

from flax import linen as nn
import jax
import jax.numpy as jnp

from geometry import (conjugate, exp_rotation, heading, invariant_state, multiply,
                      normalize_context, rotate)
from koopman import Koopman, KoopmanConfig
from normalization import unit_quaternion
from robot_config import RobotConfig


def quaternion_from_6d(features):
    """Project two column vectors onto SO(3), with finite degenerate fallbacks."""
    def normalize(v, fallback):
        squared = (v*v).sum(-1, keepdims=True)
        return jnp.where(squared > 1e-10, v / jnp.sqrt(jnp.maximum(squared, 1e-10)), fallback)
    x = normalize(features[..., :3], jnp.array([1., 0., 0.]))
    axis = jax.nn.one_hot(jnp.argmin(jnp.abs(x), axis=-1), 3)
    fallback = normalize(axis - (axis*x).sum(-1, keepdims=True)*x, jnp.array([0., 1., 0.]))
    y = features[..., 3:6]
    y = normalize(y - (y*x).sum(-1, keepdims=True)*x, fallback)
    z = jnp.cross(x, y)
    # Select the largest quaternion component for a stable matrix conversion.
    m00,m10,m20 = (x[..., i] for i in range(3))
    m01,m11,m21 = (y[..., i] for i in range(3))
    m02,m12,m22 = (z[..., i] for i in range(3))
    t = jnp.stack((1+m00+m11+m22, 1+m00-m11-m22,
                   1-m00+m11-m22, 1-m00-m11+m22), -1)
    candidates = jnp.stack((jnp.stack((t[...,0],m21-m12,m02-m20,m10-m01),-1),
                            jnp.stack((m21-m12,t[...,1],m01+m10,m02+m20),-1),
                            jnp.stack((m02-m20,m01+m10,t[...,2],m12+m21),-1),
                            jnp.stack((m10-m01,m02+m20,m12+m21,t[...,3]),-1)), -2)
    candidates /= 2*jnp.sqrt(jnp.maximum(t[..., :, None], 1e-8))
    q = jnp.take_along_axis(candidates, jnp.argmax(t, -1)[...,None,None], -2)[...,0,:]
    return unit_quaternion(q)


def reconstruct(previous, features, dt):
    """Use predicted height/tilt/joints; integrate XY and yaw from velocities."""
    q = unit_quaternion(previous[...,3:7])
    integrated = multiply(q, exp_rotation(.5 * dt * (previous[...,39:42] + features[...,10:13])))
    tilt = quaternion_from_6d(features[...,1:7])
    tilt = multiply(conjugate(heading(tilt)), tilt)
    following_q = unit_quaternion(multiply(heading(integrated), tilt))
    velocity = rotate(following_q, features[...,7:10])
    position = previous[...,:3] + .5 * dt * (previous[...,36:39] + velocity)
    position = position.at[...,2].set(features[...,0])
    return jnp.concatenate((position, following_q, features[...,13:42], velocity,
                            features[...,10:13], features[...,42:71]), -1)


class KoopmanWorldModel(nn.Module):
    robot_config: RobotConfig
    config: KoopmanConfig = KoopmanConfig()

    def setup(self):
        if (self.config.state_dim, self.config.action_dim, self.config.context_dim) != (71,29,33) or (self.robot_config.state_dim,self.robot_config.action_dim,self.robot_config.context_dim) != (71,29,33):
            raise ValueError('G1 physical adapter expects 71 state features, 29 actions, and 33 context values; use Koopman directly for other dimensions')
        self.core = Koopman(self.config)

    def state_features(self, state, normalization):
        features = invariant_state(state)
        return features if normalization is None else (features-normalization.state.mean)/normalization.state.scale

    def encode_state(self, state, normalization=None):
        return self.core.encode(self.state_features(state, normalization))

    def advance_latent(self, latent, action, context):
        return self.core.advance(latent, action, normalize_context(context))

    def decode_state(self, latent, previous, normalization=None):
        features = self.core.decode(latent)
        if normalization is not None:
            features = features*normalization.state.scale + normalization.state.mean
        return reconstruct(previous, features, self.config.dt)

    def __call__(self, state, action, context, normalization=None):
        latent = self.advance_latent(self.encode_state(state, normalization), action, context)
        return self.decode_state(latent, state, normalization)

    def rollout_latent(self, state, actions, context, normalization=None):
        return self.core.rollout(self.state_features(state, normalization), actions, normalize_context(context))

    def rollout(self, state, actions, context, normalization=None):
        def step(carry, inputs):
            z, physical = carry
            z = self.advance_latent(z, *inputs)
            physical = self.decode_state(z, physical, normalization)
            return (z, physical), (physical, z)
        _, (physical, latent) = jax.lax.scan(step, (self.encode_state(state, normalization), state),
            (jnp.swapaxes(actions,0,1), jnp.swapaxes(context,0,1)))
        return jnp.swapaxes(physical,0,1), jnp.swapaxes(latent,0,1)
