from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


class G1Simulation:
    """Load the G1 model and advance independent worlds with MJX Warp."""

    def __init__(self, num_envs=16, decimation=4):
        if not isinstance(decimation, int) or decimation < 1:
            raise ValueError("decimation must be a positive integer")
        self.num_envs = num_envs
        self.decimation = decimation
        path = Path(__file__).parent / "unitree_g1/scene_29dof_mjlab.xml"
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.mjx_model = mjx.put_model(self.model, impl="warp")

        def make_env(i):
            data = mjx.make_data(
                self.model, impl="warp", naconmax=128 * num_envs, njmax=250
            )
            return data.replace(qpos=data.qpos + i * 0.0)

        self.data = jax.vmap(make_env)(jnp.arange(num_envs))

        def advance(model, data):
            return jax.lax.fori_loop(
                0, self.decimation, lambda _, d: mjx.step(model, d), data
            )

        self._advance = jax.jit(jax.vmap(advance, in_axes=(None, 0)))

    def step(self, targets):
        """Apply joint targets and advance decimation physics steps."""
        targets = jnp.asarray(targets, dtype=jnp.float32)
        if targets.shape != (self.num_envs, self.model.nu):
            raise ValueError(f"Expected target shape {(self.num_envs, self.model.nu)}")
        self.data = self._advance(self.mjx_model, self.data.replace(ctrl=targets))
        return self.data
