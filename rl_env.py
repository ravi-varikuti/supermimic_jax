from simulation import G1Simulation


class G1Env:
    """Gym-style interface; implementation will be added one method at a time."""

    def __init__(self, num_envs=16, decimation=4):
        self.observation_space = None
        self.action_space = None
        self.num_envs = num_envs
        self.sim = G1Simulation(num_envs, decimation=decimation)
        self._renderer = None

    def reset(self, *, seed=None, options=None):
        """Return (observation, info)."""
        raise NotImplementedError

    def step(self, action):
        """Advance decimation substeps. action: (num_envs, 29) targets in radians.

        Returns physics data for now; Gym observations/rewards come next.
        """
        return self.sim.step(action)

    def render(self, env_index=None, port=8080):
        """Update Viser; its GUI selects one environment or shows all."""
        if env_index is not None and not 0 <= env_index < self.num_envs:
            raise ValueError("env_index is out of range")
        if self._renderer is None:
            from renderer import ViserRenderer

            self._renderer = ViserRenderer(self.sim.model, self.num_envs, port=port)
        if env_index is not None:
            self._renderer.env_selector.value = str(env_index)
        self._renderer.update(self.sim.data.qpos)

    def close(self):
        """Release simulation and viewer resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
