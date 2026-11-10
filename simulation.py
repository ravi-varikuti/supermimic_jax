from pathlib import Path
from math import isfinite

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from domain_randomization import DomainRandomization


class G1Simulation:
    """Load the G1 model and advance independent worlds with MJX Warp."""

    def __init__(self, num_envs=16, decimation=4, foot_friction=None):
        if not isinstance(decimation, int) or decimation < 1:
            raise ValueError("decimation must be a positive integer")
        self.num_envs = num_envs
        self.decimation = decimation
        path = Path(__file__).parent / "unitree_g1/scene_29dof_mjlab.xml"
        self.model = mujoco.MjModel.from_xml_path(str(path))
        if foot_friction is not None:
            if not isfinite(foot_friction) or foot_friction <= 0:
                raise ValueError("foot_friction must be positive and finite")
            for i in range(self.model.ngeom):
                if "foot" in (self.model.geom(i).name or "") and self.model.geom_group[i] == 3:
                    self.model.geom_friction[i, 0] = foot_friction
        self.mjx_model = mjx.put_model(self.model, impl="warp")
        self.randomization = DomainRandomization(self.model, self.mjx_model, num_envs)
        self.mjx_model = self.randomization.model
        model_axes = self.randomization.axes
        joint_ids = self.model.actuator_trnid[:, 0]
        self.joint_qpos_ids = self.model.jnt_qposadr[joint_ids]
        self._template = mjx.make_data(
            self.model, impl="warp", naconmax=128 * num_envs, njmax=250
        )

        def make_env(i):
            data = self._template
            return data.replace(qpos=data.qpos + i * 0.0)

        self.data = jax.vmap(make_env)(jnp.arange(num_envs))

        def advance(model, data):
            data = jax.lax.fori_loop(
                0, self.decimation, lambda _, d: mjx.step(model, d), data
            )
            return mjx.forward(model, data)

        self._advance = jax.jit(jax.vmap(advance, in_axes=(model_axes, 0)))

        def forward(model, qpos, qvel):
            fresh = self._template.replace(
                qpos=qpos, qvel=qvel, ctrl=qpos[self.joint_qpos_ids]
            )
            return mjx.forward(model, fresh)

        self._forward = jax.jit(jax.vmap(forward, in_axes=(model_axes, 0, 0)))

        def reset_one(model, old, qpos, qvel, mask):
            fresh = forward(model, qpos, qvel)
            return old.where(mask, fresh)

        self._reset = jax.jit(jax.vmap(reset_one, in_axes=(model_axes, 0, 0, 0, 0)))

    def forward(self, qpos, qvel):
        """Evaluate a reference pose/velocity without advancing the simulation."""
        return self._forward(self.mjx_model, qpos, qvel)

    def reset(self, qpos, qvel, mask=None):
        """Reset selected worlds to supplied poses/velocities with fresh buffers."""
        qpos = jnp.broadcast_to(jnp.asarray(qpos, jnp.float32), (self.num_envs, self.model.nq))
        qvel = jnp.broadcast_to(jnp.asarray(qvel, jnp.float32), (self.num_envs, self.model.nv))
        mask = jnp.ones(self.num_envs, dtype=bool) if mask is None else jnp.asarray(mask, bool)
        if mask.shape != (self.num_envs,):
            raise ValueError(f"Expected reset mask shape {(self.num_envs,)}")
        self.data = self._reset(self.mjx_model, self.data, qpos, qvel, mask)
        return self.data

    def step(self, targets):
        """Apply joint targets and advance decimation physics steps."""
        targets = jnp.asarray(targets, dtype=jnp.float32)
        if targets.shape != (self.num_envs, self.model.nu):
            raise ValueError(f"Expected target shape {(self.num_envs, self.model.nu)}")
        self.data = self._advance(self.mjx_model, self.data.replace(ctrl=targets))
        return self.data
