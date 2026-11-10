"""Live full-clip tracking and an uncorrected model rollout using identical actions."""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from configs.g1 import controller_defaults
import observations
import terminations


class FullRollout:
    def __init__(self, task, native, policy, variables, policy_norm, model, dynamics):
        self.task, self.native = task, native
        self.data = mujoco.MjData(native)
        self.nominal, self.scale = controller_defaults(native)
        default = jnp.asarray(self.nominal)
        self.observe = jax.jit(lambda q, v, a, r: observations.proprioception(
            q, v, a, default, reference_qpos=r))
        self.act = jax.jit(lambda history, goal: policy.apply(variables, history, goal, policy_norm))
        self.goal = jax.jit(lambda metadata: task.motion_goal_from_metadata(metadata, policy.config.future_frames))
        self.predict = jax.jit(lambda state, action: model.apply(
            {"params": dynamics.params}, state, action, jnp.array([[1.] + [0.] * 32]), dynamics.normalization))
        self.koopman = getattr(model.config, 'architecture', 'mlp') == 'koopman'
        if self.koopman:
            self.encode = jax.jit(lambda state: model.apply({'params': dynamics.params}, state,
                dynamics.normalization, method=model.encode_state))
            def predict_latent(z, state, action):
                z = model.apply({'params': dynamics.params}, z, action, jnp.array([[1.] + [0.] * 32]),
                                method=model.advance_latent)
                following = model.apply({'params': dynamics.params}, z, state, dynamics.normalization,
                                        method=model.decode_state)
                return z, following
            self.predict_latent = jax.jit(predict_latent)
        self.failure = jax.jit(terminations.check)

    def reset(self, motion):
        self.motion, self.steps = motion, 0
        qpos, qvel = self.task.reference.get(0, motion)
        self.origin = np.asarray(qpos[:3]).copy()
        self.origin[2] = 0.
        mujoco.mj_resetData(self.native, self.data)
        self.data.qpos[:] = np.asarray(qpos)
        self.data.qpos[:3] -= self.origin
        self.data.qvel[:] = np.asarray(qvel)
        self.data.ctrl[:] = self.data.qpos[7:]
        mujoco.mj_forward(self.native, self.data)
        self.predicted = jnp.asarray(np.r_[self.data.qpos, self.data.qvel][None], jnp.float32)
        self.latent = self.encode(self.predicted) if self.koopman else None
        self.previous = jnp.zeros((1, 29))
        self.history = observations.initialize_history(self.observe(
            jnp.asarray(self.data.qpos[None], jnp.float32), jnp.asarray(self.data.qvel[None], jnp.float32),
            self.previous, jnp.asarray(self.reference[None], jnp.float32)))
        self.model_valid, self.sim_valid = True, True
        self.first_failure = None

    @property
    def frame(self):
        return min(int(np.floor(self.steps * self.task.dt * self.task.reference.fps + 1e-5)),
                   int(self.task.reference.lengths[self.motion]) - 1)

    @property
    def reference(self):
        qpos, _ = self.task.reference.get(self.frame, self.motion)
        qpos = np.asarray(qpos).copy()
        qpos[:3] -= self.origin
        return qpos

    @property
    def finished(self):
        return self.frame >= int(self.task.reference.lengths[self.motion]) - 1 or not self.sim_valid

    def step(self):
        if self.finished:
            return
        goal = self.goal({"motion_id": jnp.array([self.motion]), "reference_frame": jnp.array([self.frame]),
                          "episode_steps": jnp.array([self.steps]), "reference_origin": jnp.asarray(self.origin[None])})
        action = jnp.clip(self.act(self.history.reshape(1, -1), goal), -20., 20.)
        if self.model_valid:
            if self.koopman:
                self.latent, predicted = self.predict_latent(self.latent, self.predicted, action)
            else:
                predicted = self.predict(self.predicted, action)
            self.model_valid = bool(jnp.isfinite(predicted).all())
            if self.model_valid:
                self.predicted = predicted
        self.data.ctrl[:] = self.nominal + self.scale * np.asarray(action[0])
        for _ in range(4):
            mujoco.mj_step(self.native, self.data)
        mujoco.mj_forward(self.native, self.data)
        self.steps += 1
        self.sim_valid = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        ref = self.reference
        self.history = observations.update_history(self.history, self.observe(
            jnp.asarray(self.data.qpos[None], jnp.float32), jnp.asarray(self.data.qvel[None], jnp.float32),
            action, jnp.asarray(ref[None], jnp.float32)))
        failed, _, _ = self.failure(jnp.asarray(self.data.qpos[None], jnp.float32),
                                    jnp.asarray(self.data.qvel[None], jnp.float32),
                                    jnp.asarray(ref[None], jnp.float32), jnp.array([self.steps]),
                                    2**30, jnp.array([self.frame]), self.task.reference.lengths[jnp.array([self.motion])])
        if bool(failed[0]) and self.first_failure is None:
            self.first_failure = self.steps
