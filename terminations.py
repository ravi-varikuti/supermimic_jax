from math import isfinite

import jax
import jax.numpy as jnp

from observations import projected_gravity


def tracking_errors(qpos, reference_qpos):
    return (jnp.abs(qpos[..., 2] - reference_qpos[..., 2]),
            jnp.abs(projected_gravity(qpos)[..., 2] - projected_gravity(reference_qpos)[..., 2]))


def probability(qpos, qvel, reference_qpos, height_width=0.025, tilt_width=0.08):
    """Smooth surrogate for the hard checks, not a formula specified by the paper.

    Each sigmoid is 0.5 at its hard threshold. Widths control the transition
    region; neither timeouts nor clip ends are physical failures.
    """
    if any(not isfinite(w) or w <= 0 for w in (height_width, tilt_width)):
        raise ValueError("Termination widths must be positive and finite")
    valid = jnp.isfinite(qpos).all(axis=-1) & jnp.isfinite(qvel).all(axis=-1)
    # Keep the forward probability finite for invalid model outputs.
    height, tilt = tracking_errors(jnp.nan_to_num(qpos), jnp.nan_to_num(reference_qpos))
    survive = (jax.nn.sigmoid((0.25 - height) / height_width)
               * jax.nn.sigmoid((0.8 - tilt) / tilt_width))
    return jnp.where(valid, 1. - survive, 1.)


def check(qpos, qvel, reference_qpos, episode_steps, max_steps,
          reference_frames, clip_lengths):
    """Separate tracking failures from time/clip truncation, one flag per world."""
    height_error, tilt_error = tracking_errors(qpos, reference_qpos)
    invalid = ~(jnp.isfinite(qpos).all(axis=-1) & jnp.isfinite(qvel).all(axis=-1))
    terminated = (height_error > 0.25) | (tilt_error > 0.8) | invalid
    timeout = episode_steps >= max_steps
    clip_end = reference_frames >= clip_lengths - 1
    return terminated, timeout | clip_end, {
        "height_error": height_error,
        "tilt_error": tilt_error,
        "invalid_state": invalid,
        "timeout": timeout,
        "clip_end": clip_end,
    }
