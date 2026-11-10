"""Shared cosine learning rate, counted in actor/world-model update blocks."""

from math import isfinite

import jax.numpy as jnp


def learning_rate_schedule(rate, decay_steps=None, updates_per_iteration=1):
    if not isfinite(rate) or rate <= 0:
        raise ValueError("Learning rate must be positive and finite")
    if decay_steps is None:
        return rate
    if decay_steps < 1 or updates_per_iteration < 1:
        raise ValueError("Schedule lengths must be positive")

    def schedule(count):
        iteration = count // updates_per_iteration
        fraction = jnp.minimum(iteration / decay_steps, 1.)
        return rate * (.03 + .97 * .5 * (1. + jnp.cos(jnp.pi * fraction)))

    return schedule
