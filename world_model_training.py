"""Autoregressive dynamics fitting, running local statistics, and clipped Adam."""

from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from normalization import Normalization, align_state
from normalization import FeatureStats
from running_norm import empty_moments, update_moments
from schedules import learning_rate_schedule


class WorldModelTrainState(train_state.TrainState):
    normalization: Normalization
    moments: object = None


class WorldModelTrainer:
    def __init__(self, model, learning_rate=2e-4, grad_clip=1.0, schedule_steps=None, updates_per_iteration=1):
        if learning_rate <= 0 or grad_clip <= 0:
            raise ValueError("learning_rate and grad_clip must be positive")
        self.model = model
        self.koopman = getattr(model.config, 'architecture', 'mlp') == 'koopman'
        rate = learning_rate_schedule(learning_rate, schedule_steps, updates_per_iteration)
        self.optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(rate))

    def init(self, key, normalization):
        """Initialize outside jit after collecting finite warm-up transitions."""
        if int(normalization.count) < 1:
            raise ValueError("Collect finite warm-up transitions before initializing training")
        robot = self.model.robot_config
        variables = self.model.init(
            key, jnp.zeros((1, robot.state_dim)), jnp.zeros((1, robot.action_dim)),
            jnp.zeros((1, robot.context_dim)), normalization,
        )
        return WorldModelTrainState.create(
            apply_fn=self.model.apply, params=variables["params"], tx=self.optimizer,
            normalization=normalization,
            moments={"state": empty_moments(71), "delta": empty_moments(70)} if self.model.config.local_frame else None,
        )

    def update_normalization(self, state, transitions):
        """Update local model moments from new simulator data only, never predictions."""
        if not self.model.config.local_frame:
            return state
        from geometry import invariant_state, local_delta
        valid = (jnp.isfinite(transitions.state).all(-1) & jnp.isfinite(transitions.next_state).all(-1)
                 & jnp.isfinite(transitions.action).all(-1) & jnp.isfinite(transitions.context).all(-1))
        current = transitions.state
        features = jnp.stack((invariant_state(current), invariant_state(transitions.next_state)), -2)
        moments = {"state": update_moments(state.moments["state"], features,
                                            jnp.broadcast_to(valid[..., None], features.shape[:-1])),
                   "delta": update_moments(state.moments["delta"], local_delta(current, transitions.next_state), valid)}
        norm = Normalization(FeatureStats(moments["state"].mean, jnp.sqrt(moments["state"].variance + .01**2)),
                             state.normalization.context,
                             FeatureStats(moments["delta"].mean, jnp.sqrt(moments["delta"].variance + 1e-4**2)),
                             moments["delta"].count)
        return state.replace(normalization=norm, moments=moments)

    def rollout(self, params, normalization, transitions):
        """Seed only from the first recorded state, then feed predictions back."""
        if transitions.state.ndim != 3 or transitions.state.shape[1] < 1:
            raise ValueError("Expected transitions shaped (batch, sequence, feature)")
        if self.koopman:
            return self.model.apply({'params': params}, transitions.state[:, 0], transitions.action,
                                    transitions.context, normalization, method=self.model.rollout)[0]

        def advance(state, inputs):
            action, context = inputs
            following = self.model.apply({"params": params}, state, action, context, normalization)
            return following, following

        inputs = (jnp.swapaxes(transitions.action, 0, 1),
                  jnp.swapaxes(transitions.context, 0, 1))
        _, predictions = jax.lax.scan(advance, transitions.state[:, 0], inputs)
        return jnp.swapaxes(predictions, 0, 1)

    def loss(self, params, normalization, transitions):
        if self.koopman:
            total, terms = self.koopman_loss(params, normalization, transitions)
            return total, terms['per_step_loss']
        predictions = self.rollout(params, normalization, transitions)
        if self.model.config.local_frame:
            from geometry import local_delta
            floor = jnp.array([.003]*3 + [.005]*3 + [.02]*3 + [.05]*3 + [.003]*29 + [.05]*29)
            error = local_delta(transitions.next_state, predictions) / jnp.maximum(normalization.delta.scale, floor)
            per_step = jnp.mean(error ** 2, axis=(0, 2))
            return per_step.mean(), per_step
        targets = align_state(transitions.next_state, predictions, self.model.robot_config)
        stats = normalization.delta if self.model.config.predict_delta else normalization.state
        error = (predictions - targets) / stats.scale
        per_step = jnp.mean(error ** 2, axis=(0, 2))
        return per_step.mean(), per_step

    def koopman_loss(self, params, normalization, transitions):
        """Teacher-forced one-step + open-loop state loss; optional latent loss."""
        model, c = self.model, self.model.config
        apply = lambda method, *args: model.apply({'params': params}, *args, method=method)
        target = apply(model.state_features, transitions.next_state, normalization)
        current = apply(model.encode_state, transitions.state, normalization)
        one = apply(model.advance_latent, current, transitions.action, transitions.context)
        projected, latent = apply(model.rollout_latent, transitions.state[:, 0], transitions.action,
                                  transitions.context, normalization)
        one_loss = jnp.mean((one[..., :c.state_dim] - target)**2)
        per_step = jnp.mean((projected-target)**2, axis=(0,2))
        latent_loss = jnp.array(0., jnp.float32)
        if c.latent_weight:
            encoded_target = apply(model.encode_state, transitions.next_state, normalization)
            latent_loss = jnp.mean((latent[..., c.state_dim:] - encoded_target[..., c.state_dim:])**2)
        total = c.one_step_weight*one_loss + c.multi_step_weight*per_step.mean() + c.latent_weight*latent_loss
        return total, {'one_step_loss': one_loss, 'multi_step_loss': per_step.mean(),
                       'latent_loss': latent_loss, 'per_step_loss': per_step}

    def update(self, state, sample):
        """Pure update; skip unready samples and nonfinite loss/gradients."""
        def train():
            loss_fn = self.koopman_loss if self.koopman else self.loss
            (loss, details), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, state.normalization, sample.transitions,
            )
            per_step = details['per_step_loss'] if self.koopman else details
            norm = optax.global_norm(gradients)
            finite = jnp.isfinite(loss) & jnp.isfinite(norm)
            updated = jax.lax.cond(finite, lambda: state.apply_gradients(grads=gradients), lambda: state)
            metrics = {"loss": loss, "per_step_loss": per_step,
                       "grad_norm": norm, "updated": finite}
            if self.koopman:
                metrics.update(details)
            return updated, metrics

        def skip():
            metrics = {"loss": jnp.array(0., jnp.float32),
                           "per_step_loss": jnp.zeros(sample.transitions.state.shape[1], jnp.float32),
                           "grad_norm": jnp.array(0., jnp.float32), "updated": jnp.array(False)}
            if self.koopman:
                metrics.update({name: jnp.array(0., jnp.float32) for name in
                                ('one_step_loss', 'multi_step_loss', 'latent_loss')})
            return state, metrics

        return jax.lax.cond(sample.valid, train, skip)
