"""Robot dimensions shared by state encoding and future training modules."""

from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class RobotConfig:
    # Each entry is (field name, shape within one environment). () means scalar.
    state_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    context_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    action_dim: int
    observation_dim: int
    history_length: int = 10
    quaternion_slices: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        if min(self.action_dim, self.observation_dim, self.history_length) < 1:
            raise ValueError("Action, observation, and history dimensions must be positive")
        if not self.state_shapes:
            raise ValueError("At least one state field is required")
        for fields in (self.state_shapes, self.context_shapes):
            if len({name for name, _ in fields}) != len(fields):
                raise ValueError("Field names must be unique within state/context")
            if any(dim < 1 for _, shape in fields for dim in shape):
                raise ValueError("Field dimensions must be positive")
        used = set()
        for start, end in self.quaternion_slices:
            if start < 0 or end - start != 4 or end > self.state_dim:
                raise ValueError("Quaternion slices must select four packed state values")
            if used.intersection(range(start, end)):
                raise ValueError("Quaternion slices must not overlap")
            used.update(range(start, end))

    @property
    def state_dim(self):
        return sum(prod(shape) for _, shape in self.state_shapes)

    @property
    def context_dim(self):
        return sum(prod(shape) for _, shape in self.context_shapes)

    @property
    def policy_observation_dim(self):
        return self.observation_dim * self.history_length
