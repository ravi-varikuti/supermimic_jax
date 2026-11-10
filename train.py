"""Single-GPU collect -> fit dynamics -> improve actor/critic -> evaluate/save."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from checkpoints import checkpoint_metadata, load_checkpoint, save_checkpoint
from collector import CollectorState, PolicyCollector
from critic import ValueCritic
from evaluation import Evaluator
from normalization import FeatureStats, Normalization
from policy import MotionPolicy, PolicyConfig
from policy_training import PolicyTrainer, PolicyTrainingConfig
from reference import DEFAULT_MOTION
from replay_buffer import ReplayBuffer
from reset_sampling import ReturnSampler
from rl_env import G1Env
from train_config import RunConfig, preset
from transitions import make_transition
from world_model import build_world_model
from world_model_training import WorldModelTrainer
from wandb_logging import WandbLogger


class RunState(NamedTuple):
    collector: object
    replay: object
    dynamics: object
    actor_critic: object
    key: object
    iteration: object
    simulator_steps: object
    update_blocks: object


def asset_fingerprint(motion_path):
    """Hash source, active robot assets and motion bytes to reject incompatible resumes."""
    root = Path(__file__).resolve().parent
    motion_path = Path(motion_path)
    motions = sorted(motion_path.glob("*.npz")) if motion_path.is_dir() else [motion_path]
    files = sorted(root.glob("*.py")) + sorted((root / "configs").glob("*.py"))
    files += sorted((root / "unitree_g1").rglob("*.xml")) + motions
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class TrainingRun:
    def __init__(self, config, motion_path):
        self.config = c = config
        self.motion_path = motion_path
        self.env = env = G1Env(c.num_envs, motion_path=motion_path, episode_length_s=c.episode_length_s)
        self.sampler = ReturnSampler(np.asarray(env.reference.lengths), env.reference.fps,
                                     c.num_envs, env.max_episode_steps, max_reward=5. * env.dt)
        key, reset_key, actor_key, dynamics_key, critic_key, collector_key = jax.random.split(jax.random.key(c.seed), 6)
        ids, frames = self.sampler.sample_jax(reset_key, self.sampler.probabilities(), c.num_envs)
        env.reset(seed=c.seed, motion_id=np.asarray(ids), frame=np.asarray(frames))
        self.policy = policy = MotionPolicy(env.robot_config, PolicyConfig(
            encoder_hidden=c.policy_hidden, decoder_hidden=c.policy_hidden,
            goal_frame_dim=73, bound_actions=False))
        variables = policy.init(actor_key, env.observation_history.reshape(c.num_envs, -1),
                                env.motion_goal(num_frames=policy.config.future_frames))
        collector = PolicyCollector(env, policy, lambda state: env.motion_goal(state, policy.config.future_frames),
                                    env.replay_metadata, c.exploration_std, self.sampler.sample_jax)
        self.collect = jax.jit(lambda variables, carry, weights, norm: collector.collect(
            variables, carry, c.collect_steps, weights, norm))
        self.replay = replay = ReplayBuffer(env.robot_config, c.replay_capacity, c.window)
        snapshot = env.get_state()
        zeros = jnp.zeros(c.num_envs)
        example = make_transition(snapshot, snapshot, zeros, zeros.astype(bool), zeros.astype(bool),
                                  env.robot_config, metadata=env.replay_metadata(snapshot, snapshot))
        storage = replay.init(example)
        self.extend = jax.jit(replay.extend)
        self.sample_wm = jax.jit(lambda data, key: replay.sample(data, key, c.world_model_batch_size))
        self.sample_actor = jax.jit(lambda data, key: replay.sample(data, key, c.actor_batch_size))
        model = build_world_model(env.robot_config, c, dt=env.dt)
        self.wm_trainer = wm = WorldModelTrainer(model, c.learning_rate,
                                                schedule_steps=c.schedule_steps,
                                                updates_per_iteration=c.world_model_batches)
        stats = Normalization(FeatureStats(jnp.zeros(env.robot_config.state_dim), jnp.ones(env.robot_config.state_dim)),
                              FeatureStats(jnp.zeros(env.robot_config.context_dim), jnp.ones(env.robot_config.context_dim)),
                              FeatureStats(jnp.zeros(70), jnp.ones(70)),
                              jnp.array(1., jnp.float32))
        dynamics = wm.init(dynamics_key, stats)
        self.policy_trainer = actor = PolicyTrainer(
            policy, ValueCritic(env.robot_config, hidden_dims=c.critic_hidden, goal_frame_dim=73, local_frame=True), model,
            env.observation_model, env.reward_model, env.termination_probability,
            PolicyTrainingConfig(horizon=c.horizon, learning_rate=c.learning_rate,
                                 critic_epochs=c.critic_epochs, critic_minibatches=c.critic_minibatches,
                                 schedule_steps=c.schedule_steps),
        )
        self.build_batch = jax.jit(lambda sample: env.imagination_batch(sample, c.horizon, policy.config.future_frames))
        batch = self.build_batch(self.sample_actor(storage, key))
        actor_critic = actor.init(critic_key, variables, batch)
        self.state = RunState(CollectorState(env.state, collector_key), storage, dynamics, actor_critic,
                              key, np.array(0, np.int64), np.array(0, np.int64), np.array(0, np.int64))
        self.actor_update = jax.jit(actor.update)
        self.wm_update = jax.jit(wm.update)
        self.evaluator = None
        self.update_model_norm = jax.jit(wm.update_normalization)
        def update_policy_norm(state, records):
            goals = env.motion_goal_from_metadata(records.metadata, policy.config.future_frames)
            return actor.update_input_normalization(state, records.observation, goals,
                env.observation_model.noise_scale**2 / 3 if env.observation_noise else 0.)
        self.update_policy_norm = jax.jit(update_policy_norm)

    def collect_data(self):
        state = self.state
        variables = {"params": state.actor_critic.actor.params}
        carry, records = self.collect(variables, state.collector, self.sampler.probabilities(), state.actor_critic.normalization)
        storage = self.extend(state.replay, records)
        self.sampler.observe(records)
        self.state = state._replace(collector=carry, replay=storage,
                                    dynamics=self.update_model_norm(state.dynamics, records),
                                    actor_critic=self.update_policy_norm(state.actor_critic, records),
                                    simulator_steps=state.simulator_steps + records.reward.size)
        return records

    def warmup(self):
        c = self.config
        for _ in range((c.warmup_steps + c.collect_steps - 1) // c.collect_steps):
            self.collect_data()
        if not bool(self.sample_wm(self.state.replay, self.state.key).valid):
            raise RuntimeError("Warm-up has no valid replay windows; increase warmup_steps or inspect episode failures")
        for _ in range(c.warmup_updates):
            self.fit_dynamics()

    def fit_dynamics(self):
        state = self.state
        key, sample_key = jax.random.split(state.key)
        dynamics, metrics = self.wm_update(state.dynamics, self.sample_wm(state.replay, sample_key))
        self.state = state._replace(dynamics=dynamics, key=key)
        return metrics

    def iteration(self):
        records = self.collect_data()
        actor_metrics = {"actor_loss": 0., "critic_loss": 0., "actor_grad_norm": 0., "updated": False}
        for _ in range(self.config.updates_per_collect):
            for _ in range(self.config.world_model_batches):
                wm_metrics = self.fit_dynamics()
            state = self.state
            if int(state.update_blocks) >= self.config.model_only_updates:
                key, sample_key = jax.random.split(state.key)
                batch = self.build_batch(self.sample_actor(state.replay, sample_key))
                actor_critic, actor_metrics, key = self.actor_update(
                    state.actor_critic, state.dynamics, batch, key)
                state = state._replace(actor_critic=actor_critic, key=key)
            self.state = state._replace(update_blocks=state.update_blocks + 1)
        self.state = self.state._replace(iteration=self.state.iteration + 1)
        metrics = {"iteration": int(self.state.iteration), "simulator_steps": int(self.state.simulator_steps),
                   "collected_reward": float(records.reward.mean()), "episodes": int(records.done.sum()),
                   "world_model_loss": float(wm_metrics["loss"]),
                   "world_model_updated": bool(wm_metrics["updated"]),
                   "actor_updates": int(self.state.actor_critic.actor.step),
                   "world_model_updates": int(self.state.dynamics.step)}
        metrics.update(model_only=int(self.state.update_blocks) <= self.config.model_only_updates,
                       action_abs_mean=float(jnp.abs(records.action).mean()), action_abs_max=float(jnp.abs(records.action).max()))
        if self.config.world_model_type == 'koopman':
            metrics.update({f'world_model_{name}': float(wm_metrics[name]) for name in
                            ('one_step_loss', 'multi_step_loss', 'latent_loss')})
        metrics.update({name: bool(value) if name == "updated" else float(value)
                        for name, value in actor_metrics.items()})
        return {name: None if isinstance(value, float) and not np.isfinite(value) else value
                for name, value in metrics.items()}

    def evaluate(self):
        if self.evaluator is None:
            env = G1Env(self.config.eval_envs, motion_path=self.motion_path,
                        domain_randomization=False, observation_noise=False, nominal_friction=1.)
            self.evaluator = Evaluator(env, self.policy)
        return self.evaluator.evaluate({"params": self.state.actor_critic.actor.params},
                                        normalization=self.state.actor_critic.normalization)

    def save(self, path, metadata):
        save_checkpoint(path, {"run": self.state, "sampler": self.sampler.state}, metadata)

    def restore(self, path, metadata):
        loaded = load_checkpoint(path, {"run": self.state, "sampler": self.sampler.state}, metadata)
        self.state, self.sampler.state = loaded["run"], loaded["sampler"]
        self.env.set_state(self.state.collector.env_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "local", "paper"))
    parser.add_argument("--config", type=Path, help="JSON RunConfig overrides for the chosen preset")
    parser.add_argument("--motion-path", type=Path)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/g1"))
    parser.add_argument("--iterations", type=int, help="Target collection iteration, including resumed progress")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Log training and evaluation to Weights & Biases")
    parser.add_argument("--wandb-project", help="W&B project (default: supermimic)")
    parser.add_argument("--wandb-entity", help="W&B team or username")
    parser.add_argument("--wandb-name", help="W&B display name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    args = parser.parse_args()
    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume")
    if min(args.eval_every, args.checkpoint_every) < 1:
        parser.error("Evaluation/checkpoint intervals must be positive")
    if args.resume:
        if args.config or args.preset:
            parser.error("Resume reads its config from the checkpoint; omit --preset and --config")
        metadata = checkpoint_metadata(args.resume)
        config = RunConfig(**metadata["config"])
        motion_path = (args.motion_path or Path(metadata["motion_path"])).resolve()
    else:
        config = preset(args.preset or "local")
        if args.config:
            config = RunConfig(**(config.to_dict() | json.loads(args.config.read_text())))
        motion_path = (args.motion_path or DEFAULT_MOTION).resolve()
        metadata = {"config": config.to_dict(), "motion_path": str(motion_path),
                    "fingerprint": asset_fingerprint(motion_path), "representation_version": 2}
    if metadata["fingerprint"] != asset_fingerprint(motion_path):
        raise ValueError("Source or motion assets changed since this checkpoint")
    iterations = args.iterations if args.iterations is not None else (2 if args.preset == "smoke" else 320000 // config.updates_per_collect)
    if iterations < 0:
        parser.error("iterations must be nonnegative")
    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        parser.error("Select exactly one GPU, e.g. CUDA_VISIBLE_DEVICES=0 python train.py ...")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.run_dir / "latest.ckpt").exists() and not args.resume:
        parser.error("Run directory already contains a checkpoint; use --resume or another directory")
    json_write(args.run_dir / "config.json", metadata)
    run = TrainingRun(config, motion_path)
    if args.resume:
        run.restore(args.resume, metadata)
        print(f"Resumed collection iteration {int(run.state.iteration)}", flush=True)

    logger = WandbLogger(args.run_dir, metadata, enabled=args.wandb,
                         project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, mode=args.wandb_mode)
    succeeded = False
    try:
        if logger.identity is not None:
            metadata["wandb"] = logger.identity
        json_write(args.run_dir / "config.json", metadata)
        train_loop(args, run, metadata, iterations, logger)
        succeeded = True
    except KeyboardInterrupt:
        run.save(args.run_dir / "latest.ckpt", metadata)
        print("Interrupted; saved latest.ckpt", flush=True)
    finally:
        logger.finish(exit_code=0 if succeeded else 1)


def train_loop(args, run, metadata, iterations, logger):
    if not args.resume:
        print("Collecting warm-up and fitting normalization/dynamics", flush=True)
        run.warmup()
    if args.eval_only:
        result = run.evaluate()
        json_write(args.run_dir / "evaluation.json", result)
        logger.log({"iteration": int(run.state.iteration),
                    "simulator_steps": int(run.state.simulator_steps)}, result)
        print(json.dumps({key: value for key, value in result.items() if key != "clips"}), flush=True)
        return
    start_time = time.monotonic()
    last_eval = -1
    while int(run.state.iteration) < iterations:
        metrics = run.iteration()
        result = None
        step = int(run.state.iteration)
        metrics["elapsed_s"] = time.monotonic() - start_time
        if step % args.eval_every == 0 or step == iterations:
            result = run.evaluate()
            json_write(args.run_dir / f"evaluation_{step:07d}.json", result)
            metrics["eval_return"] = result["mean_normalized_return"]
            metrics["eval_failures"] = result["failures"]
            last_eval = step
        with (args.run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        print(json.dumps(metrics), flush=True)
        logger.log(metrics, result)
        if step % args.checkpoint_every == 0 or step == iterations:
            run.save(args.run_dir / "latest.ckpt", metadata)
    if last_eval != int(run.state.iteration):
        result = run.evaluate()
        json_write(args.run_dir / f"evaluation_{int(run.state.iteration):07d}.json", result)
        logger.log({"iteration": int(run.state.iteration),
                    "simulator_steps": int(run.state.simulator_steps)}, result)
    run.save(args.run_dir / "latest.ckpt", metadata)


if __name__ == "__main__":
    main()
