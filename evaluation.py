"""Deterministic full-clip evaluation with one rollout per motion, starting at frame 0."""

import jax
import jax.numpy as jnp
import numpy as np


def clip_metrics(reward, steps, failed, expected_steps, max_reward):
    """Normalize by the full clip, so early failures lose the remaining reward."""
    return {"return": float(reward), "normalized_return": float(reward / (expected_steps * max_reward)),
            "steps": int(steps), "expected_steps": int(expected_steps), "failed": bool(failed)}


class Evaluator:
    def __init__(self, env, policy):
        if env.domain_randomization or env.observation_noise:
            raise ValueError("Evaluation requires domain randomization and sensor noise disabled")
        self.env, self.policy = env, policy
        rate = env.dt * env.reference.fps
        self.expected_steps = np.ceil((np.asarray(env.reference.lengths) - 1) / rate - 1e-7).astype(int)
        # Full clips, independent of the shorter training timeout.
        env.max_episode_steps = int(self.expected_steps.max()) + 1
        self._run = jax.jit(self._group)

    def _group(self, variables, initial, motion_ids, valid, normalization=None):
        env = self.env
        state, _, _ = env.reset_jax(initial, motion_ids=motion_ids, frames=0)
        zero = jnp.zeros(env.num_envs, jnp.float32)
        carry = (state, valid, zero, jnp.zeros(env.num_envs, jnp.int32),
                 jnp.zeros(env.num_envs, bool), zero, zero)

        def advance(carry):
            state, active, total, steps, failed, root_error, joint_error = carry
            args = (variables, state.observation_history.reshape(env.num_envs, -1),
                    env.motion_goal(state, self.policy.config.future_frames))
            action = self.policy.apply(*args) if normalization is None else self.policy.apply(*args, normalization)
            following, _, reward, terminated, truncated, _ = env.step_jax(state, action)
            reference, _ = env._reference(following)
            root = jnp.sum((following.data.qpos[:, :3] - reference[:, :3]) ** 2, axis=-1)
            joints = jnp.mean((following.data.qpos[:, 7:] - reference[:, 7:]) ** 2, axis=-1)
            total += jnp.where(active & jnp.isfinite(reward), reward, 0.)
            steps += active.astype(jnp.int32)
            failed |= active & terminated
            root_error += jnp.where(active, root, 0.)
            joint_error += jnp.where(active, joints, 0.)
            active &= ~(terminated | truncated)
            # Keep finished worlds in valid initial poses while others finish.
            following, _, _ = env.reset_jax(following, ~active, motion_ids=motion_ids, frames=0)
            return following, active, total, steps, failed, root_error, joint_error

        result = jax.lax.while_loop(lambda c: jnp.any(c[1]), advance, carry)
        return result[2:]

    def evaluate(self, variables, motion_ids=None, normalization=None):
        env = self.env
        ids = list(range(env.reference.num_motions)) if motion_ids is None else list(motion_ids)
        if not ids or any(i < 0 or i >= env.reference.num_motions for i in ids):
            raise ValueError("Evaluation needs valid motion IDs")
        rows = []
        for start in range(0, len(ids), env.num_envs):
            group = ids[start:start + env.num_envs]
            padded = np.array(group + [group[-1]] * (env.num_envs - len(group)), np.int32)
            valid = jnp.arange(env.num_envs) < len(group)
            returns, steps, failed, root_error, joint_error = jax.device_get(
                self._run(variables, env.state, jnp.asarray(padded), valid, normalization))
            for slot, motion in enumerate(group):
                row = clip_metrics(returns[slot], steps[slot], failed[slot], self.expected_steps[motion], 5. * env.dt)
                root_rmse = float(np.sqrt(root_error[slot] / max(steps[slot], 1)))
                joint_rmse = float(np.sqrt(joint_error[slot] / max(steps[slot], 1)))
                row.update(motion_id=motion, motion=env.reference.motion_names[motion],
                           root_position_rmse=root_rmse if np.isfinite(root_rmse) else None,
                           joint_position_rmse=joint_rmse if np.isfinite(joint_rmse) else None)
                rows.append(row)
        return {"num_motions": len(rows), "failures": sum(r["failed"] for r in rows),
                "mean_normalized_return": float(np.mean([r["normalized_return"] for r in rows])),
                "clips": rows}
