"""Compare autoregressive world-model predictions with Warp replay and fresh MuJoCo."""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from configs.g1 import G1_CONFIG, controller_defaults
from normalization import align_state
import observations
from reference import ReferenceMotion
from renderer import ViserRenderer
from replay_buffer import ReplayBuffer
from reward_model import TrackingRewardModel
import rewards
from rl_env import G1Env
import terminations
from view_checkpoint import load_actor
from full_rollout import FullRollout


def predict(model, dynamics, initial, actions, context):
    if getattr(model.config, 'architecture', 'mlp') == 'koopman':
        contexts = jnp.broadcast_to(context[:, None], actions.shape[:2] + (context.shape[-1],))
        following, _ = model.apply({'params': dynamics.params}, initial, actions, contexts,
                                    dynamics.normalization, method=model.rollout)
        return jnp.concatenate((initial[:, None], following), 1)
    def advance(state, action):
        following = model.apply({"params": dynamics.params}, state, action, context, dynamics.normalization)
        return following, following
    _, following = jax.lax.scan(advance, initial, jnp.swapaxes(actions, 0, 1))
    return jnp.concatenate((initial[:, None], jnp.swapaxes(following, 0, 1)), 1)


def summarize(truth, predicted, actual_reward, predicted_reward, failure_step, normalization):
    report = {}
    for horizon in (8, 32):
        expected = np.asarray(align_state(jnp.asarray(truth[:, horizon]), jnp.asarray(predicted[:, horizon]), G1_CONFIG))
        error = predicted[:, horizon] - expected
        finite = np.isfinite(predicted[:, :horizon + 1]).all(axis=(1, 2))
        normalized_error = error
        if normalization.delta.scale.shape[-1] == 70:
            from geometry import local_delta
            normalized_error = np.asarray(local_delta(jnp.asarray(expected), jnp.asarray(predicted[:, horizon])))
        report[str(horizon)] = {
            "samples": len(truth), "finite_predictions": int(finite.sum()),
            "sim_failure_before_horizon": int((failure_step < horizon).sum()),
            "root_position_error_m": float(np.linalg.norm(error[:, :3], axis=-1).mean()),
            "joint_position_rmse_rad": float(np.sqrt(np.mean(error[:, 7:36] ** 2))),
            "joint_velocity_rmse_rad_s": float(np.sqrt(np.mean(error[:, 42:] ** 2))),
            "normalized_state_mse": float(np.mean((normalized_error / np.asarray(normalization.delta.scale)) ** 2)),
            "actual_reward_sum_mean": float(actual_reward[:, :horizon].sum(1).mean()),
            "predicted_reward_sum_mean": float(predicted_reward[:, :horizon].sum(1).mean()),
        }
    return report


def fresh_rollouts(task, native_model, policy, variables, policy_normalization=None):
    """Each clip starts at frame zero. Continue 32 steps without reset for diagnosis.

    Actions come from actual simulated observations and are shared by the model.
    Steps after the first failure are explicitly marked in the results/viewer.
    """
    n, horizon = task.reference.num_motions, 32
    truth = np.zeros((n, horizon + 1, 71), np.float32)
    actions = np.zeros((n, horizon, 29), np.float32)
    origins = np.zeros((n, 3), np.float32)
    act = jax.jit(policy.apply)
    local = policy.robot_config.observation_dim == 99
    nominal, scale = controller_defaults(native_model) if local else (native_model.qpos0[7:], np.full(29, .25))
    default = jnp.asarray(nominal, jnp.float32)
    observe = jax.jit(lambda q, v, a, ref: observations.proprioception(q, v, a, default, reference_qpos=ref if local else None))
    goal_fn = jax.jit(lambda m: task.motion_goal_from_metadata(m, policy.config.future_frames))
    data = mujoco.MjData(native_model)
    for motion in range(n):
        qpos, qvel = task.reference.get(0, motion)
        origin = np.asarray(qpos[:3]).copy()
        origin[2] = 0.
        origins[motion] = origin
        mujoco.mj_resetData(native_model, data)
        data.qpos[:] = np.asarray(qpos)
        data.qpos[:3] -= origin
        data.qvel[:] = np.asarray(qvel)
        data.ctrl[:] = data.qpos[7:]
        mujoco.mj_forward(native_model, data)
        previous = jnp.zeros((1, 29))
        history = observations.initialize_history(observe(jnp.asarray(data.qpos[None], jnp.float32),
                                                          jnp.asarray(data.qvel[None], jnp.float32), previous,
                                                          jnp.asarray(qpos[None])))
        truth[motion, 0] = np.r_[data.qpos, data.qvel]
        for step in range(horizon):
            frame = int(np.floor(step * task.dt * task.reference.fps + 1e-5))
            goal = goal_fn({"motion_id": jnp.array([motion]), "reference_frame": jnp.array([frame]),
                            "episode_steps": jnp.array([step]), "reference_origin": jnp.asarray(origin[None])})
            action = np.asarray(act(variables, history.reshape(1, -1), goal, policy_normalization))[0]
            action = np.clip(action, -20., 20.) if local else action
            actions[motion, step] = action
            data.ctrl[:] = nominal + scale * action
            for _ in range(4):
                mujoco.mj_step(native_model, data)
            mujoco.mj_forward(native_model, data)
            truth[motion, step + 1] = np.r_[data.qpos, data.qvel]
            previous = jnp.asarray(action[None])
            ref, _ = task.reference.get(int(np.floor((step + 1) * task.dt * task.reference.fps + 1e-5)), motion)
            history = observations.update_history(history, observe(jnp.asarray(data.qpos[None], jnp.float32),
                                                                     jnp.asarray(data.qvel[None], jnp.float32), previous,
                                                                     jnp.asarray(ref[None])))
    frames = np.floor(np.arange(1, horizon + 1) * task.dt * task.reference.fps + 1e-5).astype(np.int32)
    metadata = {"motion_id": jnp.broadcast_to(jnp.arange(n)[:, None], (n, horizon)),
                "next_reference_frame": jnp.broadcast_to(jnp.asarray(frames), (n, horizon)),
                "reference_origin": jnp.broadcast_to(jnp.asarray(origins[:, None]), (n, horizon, 3))}
    reference = task.reference_from_metadata(metadata, next_frame=True)
    terminated, _, _ = terminations.check(jnp.asarray(truth[:, 1:, :36]), jnp.asarray(truth[:, 1:, 36:]),
                                         reference[..., :36], jnp.zeros((n, horizon)), 100000,
                                         metadata["next_reference_frame"], task.reference.lengths[:, None])
    terminated = np.asarray(terminated)
    first_failure = np.where(terminated.any(1), terminated.argmax(1) + 1, horizon + 1)
    context = np.zeros((n, 33), np.float32)
    context[:, 0] = 1.
    previous = np.concatenate((np.zeros((n, 1, 29), np.float32), actions[:, :-1]), 1)
    return truth, actions, context, previous, reference, first_failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/g1/latest.ckpt")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/g1/model_diagnostics.json")
    args = parser.parse_args()
    policy, variables, config, metadata, iteration, model, dynamics, storage = load_actor(args.checkpoint, diagnostic_data=True)
    print(f"Loaded checkpoint iteration {iteration:,}", flush=True)
    task = G1Env.__new__(G1Env)
    task.robot_config, task.dt = policy.robot_config, .02
    task.reference = ReferenceMotion(metadata["motion_path"])
    native = mujoco.MjModel.from_xml_path(str(ROOT / "unitree_g1/scene_29dof_mjlab.xml"))
    for i in range(native.ngeom):
        if "foot" in (native.geom(i).name or "") and native.geom_group[i] == 3:
            native.geom_friction[i, 0] = 1.
    anchor = "pelvis" if metadata.get("representation_version", 1) >= 2 else "torso_link"
    reward_fn = jax.jit(TrackingRewardModel(native, policy.robot_config, rewards.TRACKED_BODIES, anchor, task.dt))
    rollout_fn = jax.jit(lambda initial, action, context: predict(model, dynamics, initial, action, context))
    replay = ReplayBuffer(G1_CONFIG, config.replay_capacity, sequence_length=32)
    sample = jax.jit(lambda s: replay.sample(s, jax.random.key(41), args.samples))(storage)
    if not bool(sample.valid):
        raise RuntimeError("No uninterrupted 32-step replay sequences are available")
    records = sample.transitions
    initial = records.state[:, 0]
    true_states = jnp.concatenate((initial[:, None], records.next_state), 1)
    predicted = rollout_fn(initial, records.action, records.context[:, 0])
    reference = task.reference_from_metadata(records.metadata, next_frame=True)
    reconstructed, _ = reward_fn(records.next_state, reference, records.action, records.previous_action)
    reward_error = float(jnp.abs(reconstructed - records.reward).max())
    if reward_error > 1e-4:
        raise RuntimeError(f"Checkpoint reward reconstruction mismatch: {reward_error}")
    estimated, _ = reward_fn(predicted[:, 1:], reference, records.action, records.previous_action)
    failure_steps = np.where(np.asarray(records.terminated).any(1), np.asarray(records.terminated).argmax(1) + 1, 33)
    truth_np, pred_np = np.asarray(true_states), np.asarray(predicted)
    reports = {"checkpoint_iteration": iteration, "eligible_warp_32_step_windows": int(sample.num_valid),
               "warp_reward_reconstruction_max_error": reward_error,
               "warp_replay": summarize(truth_np, pred_np, np.asarray(records.reward), np.asarray(estimated),
                                         failure_steps, dynamics.normalization)}
    sources = {"Warp replay": (truth_np, pred_np, np.asarray(records.metadata["motion_id"][:, 0]),
                                np.asarray(records.metadata["reference_frame"][:, 0]), failure_steps)}
    print("Warp replay:", json.dumps(reports["warp_replay"]), flush=True)
    print("Collecting fresh CPU MuJoCo trajectories for all clips", flush=True)
    actual, action, context, previous, reference, failures = fresh_rollouts(task, native, policy, variables, metadata["_policy_normalization"])
    predictions = rollout_fn(jnp.asarray(actual[:, 0]), jnp.asarray(action), jnp.asarray(context))
    actual_reward, _ = reward_fn(jnp.asarray(actual[:, 1:]), reference, jnp.asarray(action), jnp.asarray(previous))
    predicted_reward, _ = reward_fn(predictions[:, 1:], reference, jnp.asarray(action), jnp.asarray(previous))
    reports["fresh_native_mujoco"] = summarize(actual, np.asarray(predictions), np.asarray(actual_reward),
                                               np.asarray(predicted_reward), failures, dynamics.normalization)
    reports["notes"] = ["Warp replay is training-buffer data, not held-out data; windows do not cross resets.",
                         "Errors are measured at the specified end step; training loss averages all eight fitted steps.",
                         "Both curves use identical logged actions; predictions are autoregressive with no corrections.",
                         "Fresh data uses CPU native MuJoCo rather than the training Warp backend.",
                         "Fresh runs continue after tracking failure for diagnosis; post-failure steps may be outside training support.",
                         "Rewards are raw sums for matched actions, without survival weighting or critic bootstrap."]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    sources["Fresh CPU MuJoCo"] = (actual, np.asarray(predictions), np.arange(len(actual)), np.zeros(len(actual), int), failures)
    print("Fresh MuJoCo:", json.dumps(reports["fresh_native_mujoco"]), flush=True)
    print("Saved", args.output, flush=True)

    viewer = ViserRenderer(native, num_envs=2, port=args.port)
    viewer.show_all.value = True
    viewer.show_reference.value = False
    viewer.env_selector.visible = viewer.show_all.visible = viewer.show_reference.visible = False
    full = FullRollout(task, native, policy, variables, metadata["_policy_normalization"], model, dynamics) if metadata.get("representation_version", 1) >= 2 else None
    modes = ["Full reference tracking", "32-step model comparison"] if full is not None else ["32-step model comparison"]
    mode = viewer.server.gui.add_dropdown("Rollout mode", options=modes, initial_value=modes[0])
    motions = [f"{i}: {name}" for i, name in enumerate(task.reference.motion_names)]
    motion = viewer.server.gui.add_dropdown("Reference motion", options=motions, initial_value=motions[0])
    restart = viewer.server.gui.add_button("Restart full clip")
    next_motion = viewer.server.gui.add_checkbox("Next motion after clip", False)
    full_model = viewer.server.gui.add_checkbox("Show world model (full clip)", False)
    source = viewer.server.gui.add_dropdown("Ground truth source", options=list(sources), initial_value="Fresh CPU MuJoCo")
    chosen = viewer.server.gui.add_slider("Sample", min=0, max=len(actual) - 1, step=1, initial_value=0)
    frame = viewer.server.gui.add_slider("Step", min=0, max=32, step=1, initial_value=0)
    playing = viewer.server.gui.add_checkbox("Play", True)
    speed = viewer.server.gui.add_slider("Speed", min=.1, max=1., step=.1, initial_value=.2)
    overlay = viewer.server.gui.add_checkbox("Overlay prediction", False)
    status = viewer.server.gui.add_markdown("")
    viewer.server.gui.add_markdown(f"Checkpoint **{iteration:,}**\n\nFull reference tracking: white = simulator, cyan = reference, orange = optional model. In 32-step comparison, cyan = model.")
    viewer.server.gui.add_markdown("Fresh rollouts: the trained policy reads simulator observations and reference goals. Its actions drive both trajectories; the model feeds back its own predicted states.")
    label_sim = viewer.server.scene.add_label("/sim_label", text="Simulator truth", position=(0., 0., 1.8))
    label_model = viewer.server.scene.add_label("/model_label", text="World model", position=(2.5, 0., 1.8))
    reset_full = [True]

    @motion.on_update
    @restart.on_click
    def restart_full(_):
        reset_full[0] = True

    @mode.on_update
    def change_mode(_):
        reset_full[0] = True
        frame.value = 0

    def draw_full():
        viewer.show_all.value = full_model.value
        viewer.show_reference.value = True
        pose = full.data.qpos.copy() if full.sim_valid else full.reference
        poses = np.stack((pose, np.asarray(full.predicted[0, :36])))
        viewer.update(poses, np.repeat(full.reference[None], 2, axis=0))
        if 1 in viewer.robots:
            for _, mesh in viewer.robots[1][1]:
                mesh.color = (255, 150, 40)
        label_model.visible = full_model.value
        label_model.text = "World model (orange)"
        root_error = np.linalg.norm(pose[:3] - full.reference[:3])
        status.content = (f"**{task.reference.motion_names[full.motion]}**\n\n"
                          f"Frame **{full.frame}/{int(task.reference.lengths[full.motion]) - 1}** · t={full.steps * task.dt:.2f}s\n\n"
                          f"White = policy simulation · cyan = reference · orange = optional world model\n\n"
                          f"Simulator root tracking error **{root_error:.3f} m**. Model starts at frame zero and is never corrected."
                          + (f"\n\nTracking failure at {full.first_failure * task.dt:.2f}s; continuing clip for inspection." if full.first_failure is not None else "")
                          + ("\n\nModel became nonfinite; prediction frozen at last finite state." if not full.model_valid else "")
                          + ("\n\nSimulation became nonfinite; playback stopped." if not full.sim_valid else ""))

    @source.on_update
    def change_source(_):
        chosen.max = len(sources[source.value][0]) - 1
        chosen.value = min(chosen.value, chosen.max)
        frame.value = 0

    def draw():
        if mode.value == "Full reference tracking":
            if reset_full[0]:
                full.reset(motions.index(motion.value))
                reset_full[0] = False
            draw_full()
            return
        true, predicted, ids, starts, failures = sources[source.value]
        sample = min(int(chosen.value), len(true) - 1)
        step = int(frame.value)
        poses = np.stack((true[sample, step, :36], predicted[sample, step, :36]))
        viewer.show_all.value = not overlay.value
        viewer.show_reference.value = overlay.value
        viewer.update(poses, np.repeat(poses[1:2], 2, axis=0))
        if 1 in viewer.robots:
            for _, mesh in viewer.robots[1][1]:
                mesh.color = (0, 180, 255)
        label_model.text = "World model (cyan)"
        label_model.visible = not overlay.value
        root_error = np.linalg.norm(poses[0, :3] - poses[1, :3])
        joint_error = np.sqrt(np.mean((poses[0, 7:] - poses[1, 7:]) ** 2))
        after_failure = step > failures[sample]
        status.content = (f"**{task.reference.motion_names[int(ids[sample])]}** · start frame {int(starts[sample])}\n\n"
                          f"Step **{step}/32**, t={step * task.dt:.2f}s\n\nRoot error **{root_error:.3f} m** · joint RMSE **{joint_error:.3f} rad**"
                          + ("\n\n**After simulator tracking failure**" if after_failure else ""))

    @overlay.on_update
    def change_layout(_):
        draw()
        for client in viewer.server.get_clients().values():
            viewer._set_camera(client)

    draw()
    print(f"Comparison ready: http://127.0.0.1:{args.port}", flush=True)
    try:
        while True:
            is_full = mode.value == "Full reference tracking"
            for control in (motion, restart, next_motion, full_model):
                control.visible = is_full
            for control in (source, chosen, frame, overlay):
                control.visible = not is_full
            draw()
            time.sleep(task.dt / speed.value)
            if playing.value:
                if is_full:
                    if not full.sim_valid:
                        playing.value = False
                    elif full.finished:
                        time.sleep(.8)
                        if next_motion.value:
                            motion.value = motions[(full.motion + 1) % len(motions)]
                        reset_full[0] = True
                    else:
                        full.step()
                elif frame.value == 32:
                    time.sleep(.8)
                    frame.value = 0
                else:
                    frame.value += 1
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
