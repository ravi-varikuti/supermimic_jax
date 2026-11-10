import unittest

from flax.core import unfreeze
import jax
import jax.numpy as jnp
import numpy as np

from critic import ValueCritic
from normalization import FeatureStats, Normalization
from observation_model import ObservationModel
from policy import MotionPolicy, PolicyConfig
from policy_training import (ImaginationBatch, ImaginedRollout, PolicyTrainer,
                             PolicyTrainingConfig, lambda_returns, rollout_return)
from robot_config import RobotConfig
from world_model import WorldModel, WorldModelConfig
from world_model_training import WorldModelTrainer
import terminations


ROBOT = RobotConfig((("x", (1,)),), (), 1, 1, history_length=2)


def setup():
    batch_size, horizon = 4, 4
    batch = ImaginationBatch(
        jnp.zeros((batch_size, 1)), jnp.zeros((batch_size, 0)), jnp.zeros((batch_size, 1)),
        jnp.zeros((batch_size, 2, 1)), jnp.zeros(batch_size, jnp.int32),
        jnp.ones((batch_size, horizon + 1, 1, 1)), jnp.ones((batch_size, horizon, 1)),
        jnp.zeros((batch_size, horizon), bool), jnp.array(True),
    )
    policy = MotionPolicy(ROBOT, PolicyConfig(future_frames=1, latent_dim=2,
                                             encoder_hidden=(8,), decoder_hidden=(8,)))
    variables = policy.init(jax.random.key(0), batch.history.reshape(batch_size, -1), batch.goals[:, 0])
    model = WorldModel(ROBOT, WorldModelConfig(hidden_dims=()))
    stats = Normalization(FeatureStats(jnp.zeros(1), jnp.ones(1)),
                          FeatureStats(jnp.zeros(0), jnp.ones(0)),
                          FeatureStats(jnp.zeros(1), jnp.ones(1)), jnp.array(1))
    dynamics = WorldModelTrainer(model).init(jax.random.key(1), stats)
    params = unfreeze(dynamics.params)
    params["dynamics"]["output"]["kernel"] = jnp.array([[0.], [.2]])
    params["dynamics"]["output"]["bias"] = jnp.zeros(1)
    dynamics = dynamics.replace(params=params)
    reward = lambda state, ref, action, prev: (-((state - ref) ** 2).sum(-1) - .01 * (action ** 2).sum(-1), {})
    failure = lambda state, ref: jax.nn.sigmoid((state[..., 0] - 2.) / .1)
    observations = ObservationModel(ROBOT, lambda state, action, context: state, noise=False)
    trainer = PolicyTrainer(policy, ValueCritic(ROBOT, goal_frames=1, hidden_dims=(8,)), model,
                            observations, reward, failure,
                            PolicyTrainingConfig(horizon=horizon, learning_rate=.003,
                                                 critic_epochs=1, critic_minibatches=2))
    state = trainer.init(jax.random.key(2), variables, batch)
    # Zero initial bootstrap for a predictable toy-control objective.
    state = state._replace(critic=state.critic.replace(params=jax.tree.map(jnp.zeros_like, state.critic.params)))
    return trainer, state, dynamics, batch


class TerminationTests(unittest.TestCase):
    def test_soft_check_matches_thresholds_and_has_gradients(self):
        qpos = jnp.array([[0., 0., 1., 1., 0., 0., 0.]])
        qvel = jnp.zeros((1, 6))
        probability = lambda z: terminations.probability(qpos.at[:, 2].set(z), qvel, qpos)[0]
        self.assertLess(float(probability(1.)), .001)
        self.assertAlmostEqual(float(probability(1.25)), .5, places=4)
        self.assertGreater(float(probability(1.5)), .999)
        self.assertGreater(float(jax.grad(probability)(1.25)), 1.)
        tilted = qpos.at[:, 3:7].set(jnp.array([jnp.sqrt(.6), jnp.sqrt(.4), 0., 0.]))
        self.assertAlmostEqual(float(terminations.probability(tilted, qvel, qpos)[0]), .5, places=4)
        self.assertEqual(float(terminations.probability(qpos.at[0, 0].set(jnp.nan), qvel, qpos)[0]), 1.)
        term, trunc, _ = terminations.check(qpos, qvel, qpos, jnp.array([10]), 10,
                                           jnp.array([4]), jnp.array([5]))
        self.assertFalse(bool(term[0]))
        self.assertTrue(bool(trunc[0]))


class PolicyTrainingTests(unittest.TestCase):
    def test_survival_and_lambda_return_timing(self):
        rewards = jnp.array([[1.], [2.], [100.]])
        failures = jnp.array([[0.], [1.], [0.]])
        rollout = ImaginedRollout(None, None, rewards, failures, jnp.zeros((3, 1), bool),
                                  jnp.array([[0.], [3.], [4.], [5.]]))
        np.testing.assert_allclose(rollout_return(rollout, .9), [2.8])
        np.testing.assert_allclose(lambda_returns(rollout, .9, 1.)[:2, 0], [2.8, 2.])
        truncated = rollout._replace(termination=jnp.zeros((3, 1)),
                                     truncated=jnp.array([[True], [False], [False]]))
        np.testing.assert_allclose(rollout_return(truncated, .9), [3.7])
        self.assertAlmostEqual(float(lambda_returns(truncated, .9, .95)[0, 0]), 3.7, places=5)
        # Soft survival must remain on the actor gradient path.
        derivative = jax.grad(lambda p: rollout_return(rollout._replace(
            termination=failures.at[0, 0].set(p)), .9).sum())(.2)
        self.assertAlmostEqual(float(derivative), -1.8, places=5)

    def test_bptt_and_frozen_dynamics_parameters(self):
        trainer, state, dynamics, batch = setup()
        rollout = jax.jit(trainer.imagine)(state.actor.params, state.critic.params,
                                         dynamics.params, dynamics.normalization, batch, jax.random.key(3))
        np.testing.assert_allclose(rollout.states[1:], jnp.cumsum(.2 * rollout.actions, axis=0), atol=1e-6)
        self.assertEqual(rollout.values.shape, (5, 4))
        grads = jax.grad(lambda params: trainer.loss(state.actor.params, state.critic.params, params,
                                                     dynamics.normalization, batch, jax.random.key(3))[0])(dynamics.params)
        for value in jax.tree.leaves(grads):
            np.testing.assert_array_equal(value, 0.)
        # The final reward depends on earlier actions through the learned recurrence.
        def final_reward(bias):
            params = unfreeze(state.actor.params)
            params["decoder"]["output"]["bias"] = jnp.array([bias])
            result = trainer.imagine(params, state.critic.params, dynamics.params,
                                      dynamics.normalization, batch, jax.random.key(3))
            return result.rewards[-1].mean()
        analytic = float(jax.grad(final_reward)(0.))
        numeric = float((final_reward(.001) - final_reward(-.001)) / .002)
        self.assertGreater(abs(analytic), .1)
        self.assertAlmostEqual(analytic, numeric, places=3)

    def test_updates_improve_toy_control_and_skip_invalid_batches(self):
        trainer, state, dynamics, batch = setup()
        key = jax.random.key(4)
        fixed_critic = state.critic.params
        evaluate = jax.jit(lambda actor: trainer.loss(actor, fixed_critic, dynamics.params,
                                                      dynamics.normalization, batch, jax.random.key(5))[0])
        before = float(evaluate(state.actor.params))
        update = jax.jit(trainer.update)
        for _ in range(20):
            state, metrics, key = update(state, dynamics, batch, key)
        self.assertTrue(bool(metrics["updated"]))
        self.assertLess(float(evaluate(state.actor.params)), before * .8)
        self.assertEqual(int(state.actor.step), 20)
        self.assertEqual(int(state.critic.step), 40)
        for bad_batch in (batch._replace(valid=jnp.array(False)),
                          batch._replace(reference=batch.reference.at[0, 0, 0].set(jnp.nan))):
            skipped, metrics, _ = update(state, dynamics, bad_batch, key)
            self.assertFalse(bool(metrics["updated"]))
            for actual, expected in zip(jax.tree.leaves(skipped), jax.tree.leaves(state)):
                np.testing.assert_array_equal(actual, expected)

    def test_bootstrap_differentiates_states_but_not_critic_weights(self):
        trainer, state, dynamics, batch = setup()
        trainer.critic = ValueCritic(ROBOT, goal_frames=1, hidden_dims=())
        critic = unfreeze(trainer.critic.init(jax.random.key(8), batch.state,
                                             batch.goals[:, 0], batch.context)["params"])
        critic["value"]["output"]["kernel"] = jnp.array([[1.], [0.]])
        critic["value"]["output"]["bias"] = jnp.zeros(1)
        trainer.reward_fn = lambda state, ref, action, prev: (jnp.zeros(state.shape[:-1]), {})
        trainer.termination_fn = lambda state, ref: jnp.zeros(state.shape[:-1])

        def loss(bias, critic_params):
            actor = unfreeze(state.actor.params)
            actor["decoder"]["output"]["bias"] = jnp.array([bias])
            return trainer.loss(actor, critic_params, dynamics.params, dynamics.normalization,
                                batch, jax.random.key(9))[0]

        actor_grad, critic_grads = jax.grad(loss, argnums=(0, 1))(0., critic)
        self.assertLess(float(actor_grad), -.5)
        for value in jax.tree.leaves(critic_grads):
            np.testing.assert_array_equal(value, 0.)


if __name__ == "__main__":
    unittest.main()
