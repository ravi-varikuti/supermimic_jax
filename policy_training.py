"""Survival-weighted BPTT actor updates and fixed-target lambda-return critic fits."""

from dataclasses import dataclass
from math import isfinite
from typing import NamedTuple

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
from schedules import learning_rate_schedule
from running_norm import empty_moments, update_moments


class ImaginationBatch(NamedTuple):
    state: object                 # (B, S), replay seed only
    context: object               # (B, C), fixed within the rollout
    previous_action: object       # (B, A)
    history: object               # (B, history_length, O), clean replay history
    episode_steps: object         # (B,), for initial sensor-noise padding
    goals: object                 # (B, H+1, K, G), including terminal critic goal
    reference: object             # (B, H, S), reference at each NEXT state
    truncated: object             # (B, H), timeout or clip end at each NEXT state
    valid: object                 # scalar replay readiness
    initial_reference: object = None


class ImaginedRollout(NamedTuple):
    states: object                # (H+1, B, S)
    actions: object               # (H, B, A)
    rewards: object               # (H, B)
    termination: object           # soft failure probability (H, B)
    truncated: object             # (H, B)
    values: object                # (H+1, B)


class ActorCriticState(NamedTuple):
    actor: object
    critic: object
    normalization: object = None


class PolicyNormalization(NamedTuple):
    observation: object
    goal: object
    critic: object


@dataclass(frozen=True)
class PolicyTrainingConfig:
    horizon: int = 32
    discount: float = .99
    gae_lambda: float = .95
    learning_rate: float = 2e-4
    critic_epochs: int = 5
    critic_minibatches: int = 4
    schedule_steps: int | None = None

    def __post_init__(self):
        for value in (self.horizon, self.critic_epochs, self.critic_minibatches):
            if not isinstance(value, int) or value < 1:
                raise ValueError("Horizon, epochs and minibatches must be positive integers")
        if not 0 <= self.discount < 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid discount or lambda")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite")


def rollout_return(rollout, discount):
    """Eq. 2, including value bootstrap at time/clip truncations.

    The transition causing a failure receives its reward. Only later rewards
    and bootstraps are suppressed by that transition's failure probability.
    """
    def advance(carry, inputs):
        total, weight = carry
        reward, failure, truncated, value_next = inputs
        total = total + weight * reward
        following = weight * discount * (1. - failure)
        total = total + jnp.where(truncated, following * value_next, 0.)
        return (total, jnp.where(truncated, 0., following)), None

    zeros = jnp.zeros_like(rollout.rewards[0])
    (total, weight), _ = jax.lax.scan(
        advance, (zeros, jnp.ones_like(zeros)),
        (rollout.rewards, rollout.termination, rollout.truncated, rollout.values[1:]),
    )
    return total + weight * rollout.values[-1]


def lambda_returns(rollout, discount, gae_lambda):
    """Eq. 3: detach these targets before any critic epochs/minibatches."""
    def backward(future, inputs):
        reward, failure, truncated, value_next = inputs
        continuation = jnp.where(truncated, value_next,
                                 (1. - gae_lambda) * value_next + gae_lambda * future)
        target = reward + discount * (1. - failure) * continuation
        return target, target

    _, targets = jax.lax.scan(
        backward, rollout.values[-1],
        (rollout.rewards, rollout.termination, rollout.truncated, rollout.values[1:]),
        reverse=True,
    )
    return targets


class PolicyTrainer:
    def __init__(self, policy, critic, model, observation_model, reward_fn,
                 termination_fn, config=PolicyTrainingConfig()):
        if not policy.robot_config == critic.robot_config == model.robot_config == observation_model.config:
            raise ValueError("Actor, critic, dynamics and observations must share robot config")
        self.policy, self.critic, self.model = policy, critic, model
        self.observation_model = observation_model
        self.reward_fn, self.termination_fn, self.config = reward_fn, termination_fn, config
        self.local_frame = model.config.local_frame
        actor_rate = learning_rate_schedule(config.learning_rate, config.schedule_steps)
        critic_rate = learning_rate_schedule(config.learning_rate, config.schedule_steps,
                                            config.critic_epochs * config.critic_minibatches)
        self.actor_optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.clip(.1),
                                            optax.adam(actor_rate, b1=.9, b2=.95))
        self.critic_optimizer = optax.chain(optax.clip_by_global_norm(1.),
                                           optax.adam(critic_rate))

    def init(self, key, actor_variables, batch):
        """Keep already-collected actor weights; initialize only its optimizer and critic."""
        variables = self.critic.init(key, batch.state, batch.goals[:, 0], batch.context,
                                     reference_state=batch.initial_reference if self.local_frame else None)
        normalizers = None
        if self.local_frame:
            size = self.critic.features(batch.state, batch.goals[:, 0], batch.context,
                                        batch.initial_reference).shape[-1]
            normalizers = PolicyNormalization(empty_moments(self.policy.robot_config.observation_dim),
                                              empty_moments(batch.goals.shape[-1]), empty_moments(size))
        return ActorCriticState(
            TrainState.create(apply_fn=self.policy.apply, params=actor_variables["params"],
                              tx=self.actor_optimizer),
            TrainState.create(apply_fn=self.critic.apply, params=variables["params"],
                              tx=self.critic_optimizer),
            normalizers,
        )

    def update_input_normalization(self, state, observation, goals, noise_variance=0.):
        if state.normalization is None:
            return state
        norm = state.normalization
        norm = norm._replace(observation=update_moments(norm.observation, observation, extra_variance=noise_variance),
                             goal=update_moments(norm.goal, goals))
        return state._replace(normalization=norm)

    def imagine(self, actor_params, critic_params, model_params, normalization, batch, key, policy_normalization=None):
        h = self.config.horizon
        if batch.goals.shape[1] != h + 1 or batch.reference.shape[1] != h or batch.truncated.shape != (batch.state.shape[0], h):
            raise ValueError("Imagination task tensors must match configured horizon")
        frozen = jax.tree.map(jax.lax.stop_gradient, (model_params, critic_params, normalization, batch))
        model_params, critic_params, normalization, batch = frozen
        history = self.observation_model.init_history(batch.history, key, batch.episode_steps)
        context = batch.context
        koopman = getattr(self.model.config, 'architecture', 'mlp') == 'koopman'
        latent = self.model.apply({'params': model_params}, batch.state, normalization,
                                  method=self.model.encode_state) if koopman else None

        def advance(carry, inputs):
            state, previous_action, history, latent = carry
            goal, reference = inputs
            action = self.policy.apply({"params": actor_params}, history.observation, goal, policy_normalization)
            if self.local_frame:
                action = jnp.clip(action, -20., 20.)
            if koopman:
                latent = self.model.apply({'params': model_params}, latent, action, context,
                                           method=self.model.advance_latent)
                following = self.model.apply({'params': model_params}, latent, state, normalization,
                                              method=self.model.decode_state)
            else:
                following = self.model.apply({"params": model_params}, state, action, context, normalization)
            reward, _ = self.reward_fn(following, reference, action, previous_action)
            failure = self.termination_fn(following, reference)
            observed_reference = reference if self.policy.robot_config.observation_dim == 99 else None
            history = self.observation_model.step(history, following, action, context, observed_reference)
            return (following, action, history, latent), (following, action, reward, failure)

        _, (states, actions, rewards, failures) = jax.lax.scan(
            jax.checkpoint(advance), (batch.state, batch.previous_action, history, latent),
            (jnp.swapaxes(batch.goals[:, :-1], 0, 1), jnp.swapaxes(batch.reference, 0, 1)),
        )
        states = jnp.concatenate((batch.state[None], states), axis=0)
        # Freeze critic parameters, preserving derivatives with respect to imagined states.
        references = jnp.concatenate((batch.initial_reference[:, None], batch.reference), 1) if self.local_frame else None
        values = self.critic.apply({"params": critic_params}, states,
                                   jnp.swapaxes(batch.goals, 0, 1),
                                   jnp.broadcast_to(context, (h + 1,) + context.shape),
                                   policy_normalization.critic if policy_normalization is not None else None,
                                   jnp.swapaxes(references, 0, 1) if references is not None else None)
        return ImaginedRollout(states, actions, rewards, failures,
                               jnp.swapaxes(batch.truncated, 0, 1), values)

    def loss(self, actor_params, critic_params, model_params, normalization, batch, key, policy_normalization=None):
        rollout = self.imagine(actor_params, critic_params, model_params, normalization, batch, key, policy_normalization)
        return -rollout_return(rollout, self.config.discount).mean(), rollout

    def update(self, state, dynamics, batch, key):
        """One actor step, then shuffled critic fits. No mutation or simulator calls.

        dynamics is WorldModelTrainState. Return the advanced RNG even on skip.
        Invalid replay or nonfinite gradients leave BOTH optimizers untouched.
        """
        key, rollout_key, shuffle_key = jax.random.split(key, 3)

        def train():
            (loss, rollout), gradients = jax.value_and_grad(self.loss, has_aux=True)(
                state.actor.params, state.critic.params, dynamics.params,
                dynamics.normalization, batch, rollout_key, state.normalization,
            )
            actor_norm = optax.global_norm(gradients)
            actor = state.actor.apply_gradients(grads=gradients)
            rollout = jax.tree.map(jax.lax.stop_gradient, rollout)
            updated_norm = state.normalization
            references = None
            if self.local_frame:
                references = jnp.swapaxes(jnp.concatenate((batch.initial_reference[:, None], batch.reference), 1), 0, 1)
                goal_all = jnp.swapaxes(batch.goals, 0, 1)
                context_all = jnp.broadcast_to(batch.context, (self.config.horizon + 1,) + batch.context.shape)
                features = self.critic.features(rollout.states, goal_all, context_all, references)
                valid_states = jnp.concatenate((jnp.ones_like(rollout.truncated[:1]),
                                                 jnp.ones_like(rollout.truncated[:1]),
                                                 jnp.cumprod(~rollout.truncated[:-1], axis=0).astype(bool)), 0)
                updated_norm = state.normalization._replace(critic=update_moments(state.normalization.critic, features, valid_states))
                values = self.critic.apply({"params": state.critic.params}, rollout.states, goal_all,
                                           context_all, updated_norm.critic, references)
                rollout = rollout._replace(values=jax.lax.stop_gradient(values))
            targets = lambda_returns(rollout, self.config.discount, self.config.gae_lambda)
            # Padding after the first time/clip boundary has no critic loss.
            active = jnp.concatenate((jnp.ones_like(rollout.truncated[:1]),
                                      jnp.cumprod(~rollout.truncated[:-1], axis=0).astype(bool)), 0)
            count = targets.size
            m = self.config.critic_minibatches
            if count % m:
                raise ValueError("Batch size times horizon must be divisible by critic_minibatches")
            states = rollout.states[:-1].reshape(count, -1)
            goals = jnp.swapaxes(batch.goals[:, :-1], 0, 1).reshape((count,) + batch.goals.shape[2:])
            contexts = jnp.broadcast_to(batch.context, (self.config.horizon,) + batch.context.shape).reshape(count, batch.context.shape[-1])
            targets, weights = targets.reshape(count), active.reshape(count).astype(jnp.float32)
            flat_references = references[:-1].reshape(count, -1) if references is not None else None

            def fit_epoch(critic, epoch):
                order = jax.random.permutation(jax.random.fold_in(shuffle_key, epoch), count).reshape(m, -1)

                def fit_minibatch(critic, indices):
                    def critic_loss(params):
                        value = self.critic.apply({"params": params}, states[indices], goals[indices], contexts[indices],
                                                  updated_norm.critic if updated_norm is not None else None,
                                                  flat_references[indices] if flat_references is not None else None)
                        error = jnp.where(weights[indices] > 0, (value - targets[indices]) ** 2, 0.)
                        return error.sum() / jnp.maximum(weights[indices].sum(), 1.)
                    value, grads = jax.value_and_grad(critic_loss)(critic.params)
                    norm = optax.global_norm(grads)
                    has_data = weights[indices].sum() > 0
                    critic = jax.lax.cond(has_data, lambda: critic.apply_gradients(grads=grads), lambda: critic)
                    return critic, (value, norm)

                return jax.lax.scan(fit_minibatch, critic, order)

            critic, (critic_losses, critic_norms) = jax.lax.scan(
                fit_epoch, state.critic, jnp.arange(self.config.critic_epochs),
            )
            candidate = ActorCriticState(actor, critic, updated_norm)
            finite = jnp.isfinite(loss) & jnp.isfinite(actor_norm)
            finite &= jnp.isfinite(critic_losses).all() & jnp.isfinite(critic_norms).all()
            finite &= jnp.stack([jnp.isfinite(x).all() for x in jax.tree.leaves(candidate)]).all()
            updated = jax.lax.cond(finite, lambda: candidate, lambda: state)
            return updated, {"actor_loss": loss, "actor_grad_norm": actor_norm,
                             "critic_loss": critic_losses.mean(), "updated": finite}

        def skip():
            zero = jnp.array(0., jnp.float32)
            return state, {"actor_loss": zero, "actor_grad_norm": zero,
                           "critic_loss": zero, "updated": jnp.array(False)}

        state, metrics = jax.lax.cond(batch.valid, train, skip)
        return state, metrics, key
