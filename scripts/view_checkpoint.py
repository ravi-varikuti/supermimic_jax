"""Play a saved G1 actor in native MuJoCo + Viser while GPU training continues."""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from configs.g1 import G1_CONFIG as G1_TRACKING_CONFIG, G1_LEGACY_CONFIG as G1_CONFIG, controller_defaults
import observations
from policy import MotionPolicy, PolicyConfig
from policy_training import PolicyTrainer, PolicyTrainingConfig
from critic import ValueCritic
from renderer import ViserRenderer
from reference import ReferenceMotion
from rl_env import G1Env
import terminations
from train_config import RunConfig
from world_model import build_world_model
from world_model_training import WorldModelTrainer
from normalization import FeatureStats, Normalization
from replay_buffer import ReplayBuffer
from transitions import Transition


def named_checkpoint_tree(archive, manifest, template, prefix):
    paths, structure = jax.tree_util.tree_flatten_with_path(template)
    lookup = {item["path"]: i for i, item in enumerate(manifest["arrays"])}
    loaded = []
    for path, leaf in paths:
        name = prefix + jax.tree_util.keystr(path)
        if name not in lookup:
            raise ValueError(f"Checkpoint field missing: {name}")
        index = lookup[name]
        value = np.load(io.BytesIO(archive.read(f"arrays/{index}.npy")), allow_pickle=False)
        expected = np.asarray(leaf, np.int32) if isinstance(leaf, int) else np.asarray(leaf)
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"Checkpoint shape/dtype mismatch: {name}")
        loaded.append(jnp.asarray(value))
    return jax.tree.unflatten(structure, loaded)


def load_actor(path, *, diagnostic_data=False):
    """Read actor/critic parameters and statistics from version-1 or version-2 checkpoints.

    V1 stores RunState, then six ReturnSampler arrays. RunState ends with the
    actor/critic tree followed by RNG, iteration and simulator-step counters.
    Validate every actor/critic array against the matching optimizer template.
    Opening the ZIP once gives a consistent snapshot during atomic training saves.
    """
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("metadata.json"))
        if manifest["version"] not in (1, 2):
            raise ValueError("Unsupported checkpoint version")
        metadata = manifest["metadata"]
        config = RunConfig(**metadata["config"])
        local = metadata.get("representation_version", 1) >= 2
        robot = G1_TRACKING_CONFIG if local else G1_CONFIG
        goal_dim = 73 if local else 71
        policy = MotionPolicy(robot, PolicyConfig(encoder_hidden=config.policy_hidden,
                                                  decoder_hidden=config.policy_hidden,
                                                  goal_frame_dim=goal_dim, bound_actions=not local))
        variables = policy.init(jax.random.key(0), jnp.zeros((1, robot.policy_observation_dim)), jnp.zeros((1, 10, goal_dim)))
        trainer = PolicyTrainer(
            policy, ValueCritic(robot, hidden_dims=config.critic_hidden, goal_frame_dim=goal_dim, local_frame=local),
            build_world_model(robot, config, local_frame=local),
            SimpleNamespace(config=robot), None, None,
            PolicyTrainingConfig(horizon=config.horizon, learning_rate=config.learning_rate,
                                 critic_epochs=config.critic_epochs, critic_minibatches=config.critic_minibatches,
                                 schedule_steps=config.schedule_steps),
        )
        template = trainer.init(jax.random.key(1), variables, SimpleNamespace(
            state=jnp.zeros((1, 71)), goals=jnp.zeros((1, 1, 10, goal_dim)), context=jnp.zeros((1, 33)),
            initial_reference=jnp.zeros((1, 71))))
        leaves, structure = jax.tree.flatten(template)
        end = len(manifest["arrays"]) - 9
        start = end - len(leaves)
        if start < 0:
            raise ValueError("Invalid checkpoint layout")
        def read_tree(template, start):
            leaves, structure = jax.tree.flatten(template)
            loaded = []
            for index, leaf in enumerate(leaves, start):
                expected = np.asarray(leaf, dtype=np.int32) if isinstance(leaf, int) else np.asarray(leaf)
                spec = manifest["arrays"][index]
                if spec["shape"] != list(expected.shape) or spec["dtype"] != str(expected.dtype) or spec["key"]:
                    raise ValueError(f"Checkpoint layout mismatch at array {index}")
                loaded.append(jnp.asarray(np.load(io.BytesIO(archive.read(f"arrays/{index}.npy")), allow_pickle=False)))
            return jax.tree.unflatten(structure, loaded)

        if manifest["version"] == 2:
            trained = named_checkpoint_tree(archive, manifest, template, "['run'].actor_critic")
            iteration = int(named_checkpoint_tree(archive, manifest, np.array(0, np.int64), "['run'].iteration"))
        else:
            trained = read_tree(template, start)
            iteration = int(np.load(io.BytesIO(archive.read(f"arrays/{end + 1}.npy")), allow_pickle=False))
        metadata["_policy_normalization"] = trained.normalization
        if diagnostic_data:
            model = build_world_model(robot, config, local_frame=local)
            stats = Normalization(FeatureStats(jnp.zeros(71), jnp.ones(71)),
                                  FeatureStats(jnp.zeros(33), jnp.ones(33)),
                                  FeatureStats(jnp.zeros(70 if local else 71), jnp.ones(70 if local else 71)),
                                  jnp.array(1., jnp.float32) if local else jnp.array(1, jnp.int32))
            world_trainer = WorldModelTrainer(model, config.learning_rate,
                schedule_steps=config.schedule_steps, updates_per_iteration=config.world_model_batches)
            dynamics_template = world_trainer.init(jax.random.key(2), stats)
            dynamics_start = start - len(jax.tree.leaves(dynamics_template))
            dynamics = named_checkpoint_tree(archive, manifest, dynamics_template, "['run'].dynamics") if manifest["version"] == 2 else read_tree(dynamics_template, dynamics_start)
            n = config.num_envs
            z = lambda *shape: jnp.zeros((n,) + shape, jnp.float32)
            integers = jnp.zeros(n, jnp.int32)
            example = Transition(z(71), z(29), z(71), z(), integers.astype(bool), integers.astype(bool),
                                 z(33), z(29), z(robot.observation_dim), z(robot.observation_dim),
                                 {"motion_id": integers, "reference_frame": integers,
                                  "next_reference_frame": integers, "reference_origin": z(3),
                                  "episode_steps": integers})
            replay_template = ReplayBuffer(robot, config.replay_capacity, config.window).init(example)
            replay_start = dynamics_start - len(jax.tree.leaves(replay_template))
            storage = named_checkpoint_tree(archive, manifest, replay_template, "['run'].replay") if manifest["version"] == 2 else read_tree(replay_template, replay_start)
            return policy, {"params": trained.actor.params}, config, metadata, iteration, model, dynamics, storage
    return policy, {"params": trained.actor.params}, config, metadata, iteration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/g1/latest.ckpt")
    parser.add_argument("--motion-id", type=int, default=0)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    policy, variables, config, metadata, iteration = load_actor(args.checkpoint)
    print(f"Loaded {args.checkpoint}: collection iteration {iteration:,}", flush=True)
    # Reference/goal helpers are pure JAX and do not need a Warp simulation.
    task = G1Env.__new__(G1Env)
    local = metadata.get("representation_version", 1) >= 2
    task.robot_config = G1_TRACKING_CONFIG if local else G1_CONFIG
    task.reference = ReferenceMotion(metadata["motion_path"])
    model = mujoco.MjModel.from_xml_path(str(ROOT / "unitree_g1/scene_29dof_mjlab.xml"))
    for i in range(model.ngeom):
        if "foot" in (model.geom(i).name or "") and model.geom_group[i] == 3:
            model.geom_friction[i, 0] = 1.
    data = mujoco.MjData(model)
    task.dt = model.opt.timestep * 4
    names = task.reference.motion_names
    if not 0 <= args.motion_id < len(names):
        parser.error("motion-id is outside the reference library")
    viewer = ViserRenderer(model, num_envs=1, port=args.port)
    print(f"Policy playback: http://127.0.0.1:{args.port}", flush=True)
    options = [f"{i}: {name}" for i, name in enumerate(names)]
    motion = viewer.server.gui.add_dropdown("Motion", options=options, initial_value=options[args.motion_id])
    playing = viewer.server.gui.add_checkbox("Play", True)
    reset = viewer.server.gui.add_button("Restart motion")
    status = viewer.server.gui.add_markdown("")
    viewer.server.gui.add_markdown(f"Checkpoint iteration **{iteration:,}**\n\nCPU MuJoCo playback; deterministic actor. Training continues separately.")
    request_reset = [True]

    @motion.on_update
    @reset.on_click
    def restart(_):
        request_reset[0] = True

    nominal, scale = controller_defaults(model) if local else (model.qpos0[7:], np.full(29, .25))
    default_pose = jnp.asarray(nominal, jnp.float32)
    observe = jax.jit(lambda q, v, a, ref: observations.proprioception(q, v, a, default_pose,
                                                                       reference_qpos=ref if local else None))
    act = jax.jit(policy.apply)
    goals = jax.jit(lambda m: task.motion_goal_from_metadata(m, policy.config.future_frames))
    failure = jax.jit(terminations.check)
    last_end = ""
    try:
        while True:
            started = time.perf_counter()
            selected = options.index(motion.value)
            if request_reset[0]:
                request_reset[0] = False
                selected = options.index(motion.value)
                qpos, qvel = task.reference.get(0, selected)
                origin = np.asarray(qpos[:3]).copy()
                origin[2] = 0.
                mujoco.mj_resetData(model, data)
                data.qpos[:] = np.asarray(qpos)
                data.qpos[:3] -= origin
                data.qvel[:] = np.asarray(qvel)
                data.ctrl[:] = data.qpos[7:]
                mujoco.mj_forward(model, data)
                previous = jnp.zeros((1, model.nu), jnp.float32)
                history = observations.initialize_history(observe(jnp.asarray(data.qpos[None], jnp.float32),
                                                                  jnp.asarray(data.qvel[None], jnp.float32), previous,
                                                                  jnp.asarray(data.qpos[None], jnp.float32)))
                steps = 0
            frame = min(int(np.floor(steps * task.dt * task.reference.fps + 1e-5)), int(task.reference.lengths[selected]) - 1)
            if playing.value:
                goal = goals({"motion_id": jnp.array([selected]), "reference_frame": jnp.array([frame]),
                              "episode_steps": jnp.array([steps]), "reference_origin": jnp.asarray(origin[None])})
                action = np.asarray(act(variables, history.reshape(1, -1), goal, metadata["_policy_normalization"]))[0]
                action = np.clip(action, -20., 20.) if local else action
                data.ctrl[:] = nominal + scale * action
                for _ in range(4):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                steps += 1
                frame = min(int(np.floor(steps * task.dt * task.reference.fps + 1e-5)), int(task.reference.lengths[selected]) - 1)
                previous = jnp.asarray(action[None])
                ref, _ = task.reference.get(frame, selected)
                history = observations.update_history(history, observe(jnp.asarray(data.qpos[None], jnp.float32),
                                                                         jnp.asarray(data.qvel[None], jnp.float32), previous,
                                                                         jnp.asarray(ref[None], jnp.float32)))
            qpos, _ = task.reference.get(frame, selected)
            reference = np.asarray(qpos).copy()
            reference[:3] -= origin
            viewer.update(data.qpos[None], reference[None])
            terminated, truncated, _ = failure(jnp.asarray(data.qpos[None], jnp.float32),
                                               jnp.asarray(data.qvel[None], jnp.float32),
                                               jnp.asarray(reference[None], jnp.float32), jnp.array([steps]),
                                               int(np.ceil(config.episode_length_s / task.dt)),
                                               jnp.array([frame]), task.reference.lengths[jnp.array([selected])])
            status.content = f"Frame **{frame}** · time **{steps * task.dt:.2f}s**\n\n{last_end}"
            if playing.value and bool((terminated | truncated)[0]):
                last_end = "Last episode: tracking failure" if bool(terminated[0]) else "Last episode: clip end / timeout"
                request_reset[0] = True
                time.sleep(.5)
            time.sleep(max(0., task.dt - (time.perf_counter() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
