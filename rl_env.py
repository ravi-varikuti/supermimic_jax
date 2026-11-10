import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from gymnasium.vector.utils import batch_space

import observations
import rewards
import terminations
from configs.g1 import G1_CONFIG, controller_defaults
from env_state import EnvState
from reference import DEFAULT_MOTION, ReferenceMotion
from simulation import G1Simulation
from state_representation import pack_state, unpack_state, unpack_context
from normalization import prepare_state
from observation_model import ObservationModel
from reward_model import TrackingRewardModel
from policy_training import ImaginationBatch


class G1Env:
    """Gym-style wrappers plus pure reset_jax/step_jax for compiled rollouts."""

    def __init__(self, num_envs=16, decimation=4, motion_path=DEFAULT_MOTION,
                 episode_length_s=10.0, action_scale=None, domain_randomization=True,
                 observation_noise=True, nominal_friction=None):
        if not np.isfinite(episode_length_s) or episode_length_s <= 0:
            raise ValueError("episode_length_s must be finite and positive")
        self.num_envs = num_envs
        self.robot_config = G1_CONFIG
        self.domain_randomization = domain_randomization
        self.observation_noise = observation_noise
        self.sim = G1Simulation(num_envs, decimation=decimation, foot_friction=nominal_friction)
        nominal, motor_scale = controller_defaults(self.sim.model)
        scale = motor_scale if action_scale is None else np.broadcast_to(np.asarray(action_scale, dtype=np.float32), (self.sim.model.nu,))
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("action_scale must be positive and finite (scalar or 29 values)")
        self.action_scale = jnp.asarray(scale)
        self.reference = ReferenceMotion(motion_path)
        names = tuple(self.sim.model.joint(i).name for i in range(1, self.sim.model.njnt))
        if self.reference.joint_names != names:
            raise ValueError("Reference joint order does not match the robot")
        self.dt = self.sim.model.opt.timestep * decimation
        self.max_episode_steps = int(np.ceil(episode_length_s / self.dt))
        self.default_joint_pos = jnp.asarray(nominal)
        self.action_limit = 20.
        self.observation_model = ObservationModel(
            self.robot_config, self.observation_from_state, observations.noise_scale(True),
            noise=self.observation_noise,
        )
        zeros = jnp.zeros(num_envs, dtype=jnp.int32)
        action = jnp.zeros((num_envs, self.sim.model.nu))
        initial = observations.proprioception(
            self.sim.data.qpos, self.sim.data.qvel, action, self.default_joint_pos,
            reference_qpos=self.sim.data.qpos,
        )
        history = observations.initialize_history(initial)
        self.state = EnvState(
            self.sim.data, self.sim.mjx_model, dict(self.sim.randomization.context),
            jax.random.key(0), zeros, jnp.zeros((num_envs, 3)), zeros, zeros,
            action, history, history,
        )
        observation_size = int(np.prod(history.shape[1:]))
        self.single_observation_space = Box(-np.inf, np.inf, (observation_size,), np.float32)
        self.single_action_space = Box(-self.action_limit, self.action_limit, (self.sim.model.nu,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.joint_limits = rewards.soft_joint_limits(self.sim.model)
        self.reward_model = TrackingRewardModel(
            self.sim.model, self.robot_config, rewards.TRACKED_BODIES, "pelvis", self.dt,
        )
        self._compiled_reset = jax.jit(self.reset_jax)
        self._compiled_step = jax.jit(self.step_jax)
        self._renderer = None

    def _frames(self, state):
        elapsed = jnp.floor(state.episode_steps * (self.dt * self.reference.fps) + 1e-5)
        return jnp.minimum(state.start_frames + elapsed.astype(jnp.int32),
                           self.reference.lengths[state.motion_ids] - 1)

    def _reference(self, state):
        qpos, qvel = self.reference.get(self._frames(state), state.motion_ids)
        return qpos.at[:, :3].add(-state.reference_origins), qvel

    @property
    def reference_frames(self):
        return self._frames(self.state)

    @property
    def reference_state(self):
        return self._reference(self.state)

    def motion_goal(self, state=None, num_frames=10):
        """Future local reference features, shape (num_envs, num_frames, 73).

        Frames are at future control times t+1 .. t+num_frames and clamp at clip
        end. Legacy configurations retain their packed-state goal format.
        """
        if not isinstance(num_frames, int) or num_frames < 1:
            raise ValueError("num_frames must be a positive integer")
        state = self.state if state is None else state
        return self.motion_goal_from_metadata({
            "motion_id": state.motion_ids, "reference_frame": self._frames(state),
            "episode_steps": state.episode_steps, "reference_origin": state.reference_origins,
        }, num_frames)

    def motion_goal_from_metadata(self, metadata, num_frames=10):
        """Rebuild the same goals from replay metadata, preserving batch/time axes."""
        if not isinstance(num_frames, int) or num_frames < 1:
            raise ValueError("num_frames must be a positive integer")
        steps = metadata["episode_steps"][..., None]
        rate = self.dt * self.reference.fps
        now = jnp.floor(steps * rate + 1e-5).astype(jnp.int32)
        future = jnp.floor((steps + jnp.arange(1, num_frames + 1)) * rate + 1e-5).astype(jnp.int32)
        frames = metadata["reference_frame"][..., None] + future - now
        qpos, qvel = self.reference.get(frames, metadata["motion_id"][..., None])
        qpos = qpos.at[..., :3].add(-metadata["reference_origin"][..., None, :])
        if self.robot_config.observation_dim == 93:
            return prepare_state(pack_state({"qpos": qpos, "qvel": qvel}, self.robot_config), self.robot_config)
        from geometry import heading, conjugate, multiply, rotate, rotation6d
        current_qpos, _ = self.reference.get(metadata["reference_frame"], metadata["motion_id"])
        h_inv = conjugate(heading(current_qpos[..., 3:7]))[..., None, :]
        return jnp.concatenate((rotate(h_inv, qpos[..., :3] - current_qpos[..., None, :3]
                                      + metadata["reference_origin"][..., None, :]),
                                rotation6d(multiply(h_inv, qpos[..., 3:7])),
                                rotate(conjugate(qpos[..., 3:7]), qvel[..., :3]), qvel[..., 3:6],
                                qpos[..., 7:], .1 * qvel[..., 6:]), -1)

    def reference_from_metadata(self, metadata, *, next_frame=False):
        """Packed reference at the current or next recorded control time."""
        frames = metadata["next_reference_frame" if next_frame else "reference_frame"]
        qpos, qvel = self.reference.get(frames, metadata["motion_id"])
        qpos = qpos.at[..., :3].add(-metadata["reference_origin"])
        return prepare_state(pack_state({"qpos": qpos, "qvel": qvel}, self.robot_config),
                             self.robot_config)

    def termination_probability(self, state, reference):
        """Differentiable failure surrogate; time/clip boundaries are separate."""
        fields = unpack_state(prepare_state(state, self.robot_config), self.robot_config)
        target = unpack_state(prepare_state(reference, self.robot_config), self.robot_config)
        return terminations.probability(fields["qpos"], fields["qvel"], target["qpos"])

    def imagination_batch(self, sample, horizon=32, goal_frames=10):
        """Seed at each replay window's first state; extend reference beyond the window.

        Logged future actions/states do not enter policy optimization. Goals use
        exact control-time lookups, including noninteger reference/control rates.
        """
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        metadata = jax.tree.map(lambda value: value[:, 0], sample.transitions.metadata)
        steps = metadata["episode_steps"][:, None] + jnp.arange(horizon + 1)
        rate = self.dt * self.reference.fps
        elapsed = jnp.floor(steps * rate + 1e-5).astype(jnp.int32)
        now = jnp.floor(metadata["episode_steps"] * rate + 1e-5).astype(jnp.int32)
        frames = metadata["reference_frame"][:, None] + elapsed - now[:, None]
        extended = {"motion_id": jnp.broadcast_to(metadata["motion_id"][:, None], frames.shape),
                    "reference_frame": frames, "episode_steps": steps,
                    "reference_origin": jnp.broadcast_to(metadata["reference_origin"][:, None], frames.shape + (3,))}
        goals = self.motion_goal_from_metadata(extended, goal_frames)
        reference = self.reference_from_metadata(extended)[:, 1:]
        truncated = ((steps[:, 1:] >= self.max_episode_steps)
                     | (frames[:, 1:] >= self.reference.lengths[metadata["motion_id"]][:, None] - 1))
        initial_reference = self.reference_from_metadata(metadata)
        return ImaginationBatch(sample.transitions.state[:, 0], sample.transitions.context[:, 0],
                                sample.transitions.previous_action[:, 0], sample.history,
                                metadata["episode_steps"], goals, reference, truncated, sample.valid, initial_reference)

    @staticmethod
    def replay_metadata(before, after):
        return {
            "motion_id": before["motion_id"], "reference_frame": before["reference_frame"],
            "next_reference_frame": after["reference_frame"],
            "reference_origin": before["reference_origin"], "episode_steps": before["episode_steps"],
        }

    @property
    def motion_ids(self):
        return self.state.motion_ids

    @property
    def episode_steps(self):
        return self.state.episode_steps

    @property
    def previous_action(self):
        return self.state.previous_action

    @property
    def observation_history(self):
        return self.state.observation_history

    @property
    def start_frames(self):
        return self.state.start_frames

    @property
    def reference_origins(self):
        return self.state.reference_origins

    def _observation(self, data, action, context, key, reference_qpos):
        clean = observations.proprioception(
            data.qpos, data.qvel, action, self.default_joint_pos, context["joint_bias"],
            reference_qpos,
        )
        noisy = observations.add_noise(clean, key) if self.observation_noise else clean
        return clean, noisy

    def observation_from_state(self, state, previous_action, context, reference_state=None):
        """G1 observation adapter for predicted physical states and packed context."""
        fields = unpack_state(prepare_state(state, self.robot_config), self.robot_config)
        parameters = unpack_context(context, self.robot_config)
        reference_qpos = None
        if self.robot_config.observation_dim == 99:
            if reference_state is None:
                raise ValueError("Reference state is required for relative-orientation observations")
            reference_qpos = unpack_state(prepare_state(reference_state, self.robot_config), self.robot_config)["qpos"]
        return observations.proprioception(
            fields["qpos"], fields["qvel"], previous_action, self.default_joint_pos,
            parameters["joint_bias"], reference_qpos,
        )

    def _info(self, state):
        return {"motion_id": state.motion_ids, "reference_frame": self._frames(state),
                "domain_randomization": dict(state.context)}

    def get_state(self, state=None):
        """Replay snapshot; clean history excludes sensor noise, retains calibration.

        Use env.state itself as the carry for JIT/scan and exact simulation state.
        """
        state = self.state if state is None else state
        return {
            "qpos": state.data.qpos, "qvel": state.data.qvel,
            "joint_targets": state.data.ctrl, "previous_action": state.previous_action,
            "observation_history": state.observation_history,
            "clean_observation_history": state.clean_observation_history,
            "motion_id": state.motion_ids, "start_frame": state.start_frames,
            "reference_frame": self._frames(state), "reference_origin": state.reference_origins,
            "episode_steps": state.episode_steps, "domain_randomization": dict(state.context),
        }

    def set_state(self, state):
        """Install a compiled rollout's state for subsequent wrappers/rendering."""
        self.state = state
        self.sim.data, self.sim.mjx_model = state.data, state.model
        self.sim.randomization.model = state.model
        self.sim.randomization.context = dict(state.context)

    def reset_jax(self, state, mask=None, motion_ids=None, frames=None):
        """Pure masked reset. mask: (num_envs,) bool; overrides: scalar or full batch.

        Works inside jit/scan. Explicit clip/frame overrides must be valid.
        Returns (state, observation, info); an all-false mask is a no-op.
        """
        mask = jnp.ones(self.num_envs, bool) if mask is None else jnp.asarray(mask, bool)
        if mask.shape != (self.num_envs,):
            raise ValueError(f"mask must have shape ({self.num_envs},)")
        return jax.lax.cond(
            jnp.any(mask),
            lambda: self._reset_selected(state, mask, motion_ids, frames),
            lambda: (state, state.observation_history.reshape(self.num_envs, -1), self._info(state)),
        )

    def _reset_selected(self, state, mask, motion_ids, frames):
        key, sample_key, domain_key, noise_key = jax.random.split(state.key, 4)
        motions, starts = self.reference.sample_jax(sample_key, self.num_envs, motion_ids, frames)
        raw_qpos, _ = self.reference.get(starts, motions)
        origins = raw_qpos[:, :3].at[:, 2].set(0.)
        model, context = self.sim.randomization.reset_jax(
            state.model, state.context, mask, domain_key, self.domain_randomization,
        )
        state = state._replace(
            key=key, model=model, context=context,
            motion_ids=jnp.where(mask, motions, state.motion_ids),
            start_frames=jnp.where(mask, starts, state.start_frames),
            reference_origins=jnp.where(mask[:, None], origins, state.reference_origins),
            episode_steps=jnp.where(mask, 0, state.episode_steps),
            previous_action=jnp.where(mask[:, None], 0., state.previous_action),
        )
        qpos, qvel = self._reference(state)
        data = self.sim._reset(model, state.data, qpos, qvel, mask)
        clean, noisy = self._observation(data, state.previous_action, context, noise_key, qpos)
        state = state._replace(
            data=data,
            observation_history=jnp.where(mask[:, None, None],
                observations.initialize_history(noisy), state.observation_history),
            clean_observation_history=jnp.where(mask[:, None, None],
                observations.initialize_history(clean), state.clean_observation_history),
        )
        return state, state.observation_history.reshape(self.num_envs, -1), self._info(state)

    def reset(self, *, frame=None, motion_id=None, env_ids=None, seed=None, options=None):
        """Interactive reset wrapper; use reset_jax with a fixed-size mask in training."""
        if options is not None:
            frame, env_ids = options.get("frame", frame), options.get("env_ids", env_ids)
            motion_id = options.get("motion_id", motion_id)
        ids = np.arange(self.num_envs) if env_ids is None else np.atleast_1d(env_ids)
        if not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= self.num_envs)):
            raise ValueError("env_ids must contain valid integer environment indices")
        # Host-side validation is confined to this convenience wrapper.
        motions = None
        if motion_id is not None:
            selected = np.broadcast_to(np.asarray(motion_id), (len(ids),))
            if not np.issubdtype(selected.dtype, np.integer) or np.any((selected < 0) | (selected >= self.reference.num_motions)):
                raise ValueError("motion_id must contain valid integer clip indices")
            motions = self.state.motion_ids.at[ids].set(selected)
        frames = None
        if frame is not None:
            selected = np.broadcast_to(np.asarray(frame), (len(ids),))
            if not np.issubdtype(selected.dtype, np.integer) or np.any(selected < 0):
                raise ValueError("frame must contain nonnegative integer indices")
            frames = self.state.start_frames.at[ids].set(selected)
        state = self.state if seed is None else self.state._replace(key=jax.random.key(seed))
        # Validate frame bounds against the same seeded JAX clip samples used below.
        if frame is not None:
            sample_key = jax.random.split(state.key, 4)[1]
            chosen, _ = self.reference.sample_jax(sample_key, self.num_envs, motions, frames)
            if np.any(np.asarray(frames)[ids] >= self.reference._lengths[np.asarray(chosen)[ids]]):
                raise ValueError("frame must be valid for its selected clip")
        mask = jnp.zeros(self.num_envs, bool).at[ids].set(True)
        state, obs, info = self._compiled_reset(state, mask, motions, frames)
        self.set_state(state)
        return obs, info

    def step_jax(self, state, action):
        """Pure transition returning (state, obs, reward, terminated, truncated, info).

        No automatic reset: save the terminal transition before reset_jax.
        """
        action = jnp.asarray(action, dtype=jnp.float32)
        if action.shape != (self.num_envs, self.sim.model.nu):
            raise ValueError(f"action must have shape ({self.num_envs}, {self.sim.model.nu})")
        action = jnp.clip(action, -self.action_limit, self.action_limit)
        targets = self.default_joint_pos + self.action_scale * action + state.context["joint_bias"]
        data = self.sim._advance(state.model, state.data.replace(ctrl=targets))
        key, noise_key = jax.random.split(state.key)
        next_state = state._replace(data=data, key=key, episode_steps=state.episode_steps + 1,
                                    previous_action=action)
        qpos, qvel = self._reference(next_state)
        reference_data = self.sim._forward(state.model, qpos, qvel)
        reward, terms = rewards.tracking_reward(
            rewards.from_data(self.sim.model, data), rewards.from_data(self.sim.model, reference_data),
            action, state.previous_action, self.joint_limits, self.dt,
            anchor_index=0,
        )
        clean, noisy = self._observation(data, action, state.context, noise_key, qpos)
        next_state = next_state._replace(
            observation_history=observations.update_history(state.observation_history, noisy),
            clean_observation_history=observations.update_history(state.clean_observation_history, clean),
        )
        terminated, truncated, end_info = terminations.check(
            data.qpos, data.qvel, qpos, next_state.episode_steps, self.max_episode_steps,
            self._frames(next_state), self.reference.lengths[next_state.motion_ids],
        )
        obs = next_state.observation_history.reshape(self.num_envs, -1)
        return next_state, obs, reward, terminated, truncated, {
            **end_info, "reward_terms": terms, **self._info(next_state),
        }

    def step(self, action):
        """Gym-style wrapper: clipped normalized actions; explicit selective resets."""
        state, obs, reward, terminated, truncated, info = self._compiled_step(self.state, action)
        self.set_state(state)
        return obs, reward, terminated, truncated, info

    def render(self, env_index=None, port=8080):
        """Update Viser; its GUI selects one environment or shows all."""
        if env_index is not None and not 0 <= env_index < self.num_envs:
            raise ValueError("env_index is out of range")
        if self._renderer is None:
            from renderer import ViserRenderer

            self._renderer = ViserRenderer(self.sim.model, self.num_envs, port=port)
        if env_index is not None:
            self._renderer.env_selector.value = str(env_index)
        reference_qpos, _ = self.reference_state
        self._renderer.update(self.sim.data.qpos, reference_qpos)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
