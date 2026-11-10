"""Small single-GPU presets and a separate published-scale configuration."""

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    num_envs: int = 256
    eval_envs: int = 8
    collect_steps: int = 5
    updates_per_collect: int = 2
    replay_capacity: int = 1000
    window: int = 8
    warmup_steps: int = 100
    warmup_updates: int = 10
    model_only_updates: int = 500
    world_model_batches: int = 10
    world_model_batch_size: int = 128
    actor_batch_size: int = 256
    horizon: int = 32
    dynamics_hidden: tuple[int, ...] = (2048, 2048, 2048)
    policy_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden: tuple[int, ...] = (256, 128)
    critic_epochs: int = 5
    critic_minibatches: int = 4
    learning_rate: float = 2e-4
    schedule_steps: int = 250000
    episode_length_s: float = 10.
    exploration_std: float = .1
    world_model_type: str = 'mlp'
    koopman_features: int = 128
    koopman_encoder_hidden: tuple[int, ...] = (256, 256)
    koopman_one_step_weight: float = 1.
    koopman_multi_step_weight: float = 1.
    koopman_latent_weight: float = .1

    def __post_init__(self):
        for name in ("dynamics_hidden", "policy_hidden", "critic_hidden", "koopman_encoder_hidden"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("num_envs", "eval_envs", "collect_steps", "updates_per_collect", "replay_capacity",
                     "window", "warmup_steps", "world_model_batches", "world_model_batch_size",
                     "actor_batch_size", "horizon", "critic_epochs", "critic_minibatches", "schedule_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.warmup_updates < 0 or self.model_only_updates < 0 or self.window > self.replay_capacity:
            raise ValueError("Invalid warm-up or replay window")
        if self.actor_batch_size * self.horizon % self.critic_minibatches:
            raise ValueError("Actor batch times horizon must be divisible by critic minibatches")
        if self.replay_capacity < self.window + 9:
            raise ValueError("Replay must also retain observation history")
        if any(not isfinite(x) or x <= 0 for x in (self.learning_rate, self.episode_length_s)):
            raise ValueError("Learning rate and episode length must be positive and finite")
        if not isfinite(self.exploration_std) or self.exploration_std < 0:
            raise ValueError("Exploration must be nonnegative and finite")
        if any(not sizes or any(not isinstance(n, int) or n < 1 for n in sizes)
               for sizes in (self.dynamics_hidden, self.policy_hidden, self.critic_hidden)):
            raise ValueError("Hidden widths must be positive integers")
        if self.world_model_type not in ('mlp', 'koopman'):
            raise ValueError('world_model_type must be mlp or koopman')
        from koopman import KoopmanConfig
        KoopmanConfig(feature_dim=self.koopman_features, encoder_hidden=self.koopman_encoder_hidden,
                      one_step_weight=self.koopman_one_step_weight, multi_step_weight=self.koopman_multi_step_weight,
                      latent_weight=self.koopman_latent_weight)

    def to_dict(self):
        result = asdict(self)
        for name in ("dynamics_hidden", "policy_hidden", "critic_hidden", "koopman_encoder_hidden"):
            result[name] = list(result[name])
        return result


def preset(name):
    if name == "local":
        return RunConfig()
    if name == "smoke":
        return RunConfig(num_envs=4, eval_envs=4, collect_steps=5, updates_per_collect=1,
                         replay_capacity=64, warmup_steps=25, warmup_updates=2,
                         world_model_batches=1, world_model_batch_size=8, actor_batch_size=4,
                         horizon=4, dynamics_hidden=(32, 32), policy_hidden=(32, 32),
                         critic_hidden=(16, 8), critic_epochs=1, model_only_updates=0)
    if name == "paper":
        return RunConfig(num_envs=4096, eval_envs=64, world_model_batch_size=2048,
                         actor_batch_size=4096, dynamics_hidden=(2048, 2048, 2048),
                         policy_hidden=(2048, 1024, 512, 256))
    raise ValueError(f"Unknown preset: {name}")
