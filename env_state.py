"""Explicit environment state carried through compiled JAX rollouts."""

from typing import NamedTuple


class EnvState(NamedTuple):
    data: object
    model: object
    context: dict
    key: object
    motion_ids: object
    reference_origins: object
    start_frames: object
    episode_steps: object
    previous_action: object
    observation_history: object
    clean_observation_history: object
