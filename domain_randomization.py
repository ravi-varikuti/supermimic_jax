"""Episode-constant SuperMimic randomization, sampled and applied in JAX."""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np


MODEL_FIELDS = (
    "geom_friction", "body_ipos", "body_invweight0", "dof_invweight0",
    "actuator_acc0", "dof_M0", "light_poscom0",
)


class DomainRandomization:
    def __init__(self, native_model, model, num_envs):
        self._nominal = model
        self._torso = native_model.body("torso_link").id
        self._feet = jnp.array([
            i for i in range(native_model.ngeom)
            if "foot" in (native_model.geom(i).name or "") and native_model.geom_group[i] == 3
        ])
        self.context = {
            "friction": jnp.full(num_envs, model.geom_friction[self._feet[0], 0]),
            "joint_bias": jnp.zeros((num_envs, native_model.nu)),
            "torso_com_offset": jnp.zeros((num_envs, 3)),
        }
        self.model = model.replace(**{
            name: jnp.broadcast_to(getattr(model, name), (num_envs,) + getattr(model, name).shape)
            for name in MODEL_FIELDS
        })
        self.axes = jax.tree.map(lambda _: None, self.model).replace(
            **{name: 0 for name in MODEL_FIELDS},
        )
        # Setup only: Jacobians at qpos0. A torso COM shift changes its linear
        # Jacobian, hence M0, inverse weights, and actuator reference inertia.
        data = mujoco.MjData(native_model)
        mujoco.mj_forward(native_model, data)
        mass = np.empty((native_model.nv, native_model.nv))
        mujoco.mj_fullM(native_model, data, mass)
        jac_pos = np.zeros((native_model.nbody, 3, native_model.nv))
        jac_rot = np.zeros_like(jac_pos)
        for body in range(native_model.nbody):
            mujoco.mj_jacBodyCom(native_model, data, jac_pos[body], jac_rot[body], body)
        self._mass = jnp.asarray(mass, jnp.float32)
        self._inverse_mass = jnp.asarray(np.linalg.inv(mass), jnp.float32)
        self._jac_pos = jnp.asarray(jac_pos, jnp.float32)
        self._jac_rot = jnp.asarray(jac_rot, jnp.float32)
        self._rotation = jnp.asarray(data.xmat[self._torso].reshape(3, 3), jnp.float32)
        self._torso_mass = float(native_model.body_mass[self._torso])
        self._actuator_dofs = jnp.asarray(native_model.jnt_dofadr[native_model.actuator_trnid[:, 0]])
        self._gear = model.actuator_gear[:, 0]
        self._light_affected = jnp.asarray(native_model.light_bodyid == 0)
        self._total_mass = float(native_model.body_subtreemass[0])

    def _model_for_context(self, friction, com):
        with jax.default_matmul_precision("highest"):
            return self._constants(friction, com)

    def _constants(self, friction, com):
        """Exact G1 constants for a torso COM translation; mass/inertia stay fixed."""
        delta = self._rotation @ com
        old = self._jac_pos[self._torso]
        new = old + jnp.cross(self._jac_rot[self._torso].T, delta).T
        mass = self._mass + self._torso_mass * (new.T @ new - old.T @ old)
        # Woodbury update preserves the accurate nominal inverse and only solves
        # a 6x6 system, rather than inverting every 35x35 mass matrix in float32.
        update = jnp.concatenate((old.T, (new - old).T), axis=-1)
        eye, zero = jnp.eye(3), jnp.zeros((3, 3))
        inverse_coefficient = jnp.block([[-eye, eye], [eye, zero]]) / self._torso_mass
        product = self._inverse_mass @ update
        inverse = self._inverse_mass - product @ jnp.linalg.solve(
            inverse_coefficient + update.T @ product, product.T,
        )
        jp = self._jac_pos.at[self._torso].set(new)
        translation = jnp.einsum("bik,kl,bil->b", jp, inverse, jp) / 3
        rotation = jnp.einsum("bik,kl,bil->b", self._jac_rot, inverse, self._jac_rot) / 3
        dof_inverse = jnp.diag(inverse)
        dof_inverse = dof_inverse.at[:3].set(dof_inverse[:3].mean())
        dof_inverse = dof_inverse.at[3:6].set(dof_inverse[3:6].mean())
        return {
            "geom_friction": self._nominal.geom_friction.at[self._feet, 0].set(friction),
            "body_ipos": self._nominal.body_ipos.at[self._torso].add(com),
            "body_invweight0": jnp.stack((translation, rotation), axis=-1),
            "dof_invweight0": dof_inverse,
            "actuator_acc0": jnp.linalg.norm(inverse[:, self._actuator_dofs] * self._gear, axis=0),
            "dof_M0": jnp.diag(mass),
            "light_poscom0": self._nominal.light_poscom0 - self._light_affected[:, None] * delta * self._torso_mass / self._total_mass,
        }

    def reset_jax(self, model, context, mask, key, enabled=True):
        """Pure fixed-shape masked reset: no CPU work or mutation."""
        count = mask.shape[0]
        friction_key, bias_key, com_key = jax.random.split(key, 3)
        friction = jax.random.uniform(friction_key, (count,), minval=.3, maxval=1.2)
        bias = jax.random.uniform(bias_key, (count, self._nominal.nu), minval=-.01, maxval=.01)
        com = jax.random.uniform(com_key, (count, 3), minval=-1., maxval=1.) * jnp.array([.025, .05, .05])
        friction = jnp.where(enabled, friction, self._nominal.geom_friction[self._feet[0], 0])
        bias, com = jnp.where(enabled, bias, 0.), jnp.where(enabled, com, 0.)
        values = jax.vmap(self._model_for_context)(friction, com)
        model = model.replace(**{
            name: jnp.where(mask.reshape((-1,) + (1,) * (value.ndim - 1)), value, getattr(model, name))
            for name, value in values.items()
        })
        sampled = {"friction": friction, "joint_bias": bias, "torso_com_offset": com}
        context = {
            name: jnp.where(mask.reshape((-1,) + (1,) * (value.ndim - 1)), value, context[name])
            for name, value in sampled.items()
        }
        return model, context
