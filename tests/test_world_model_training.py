import unittest

from flax.core import unfreeze
import jax
import jax.numpy as jnp
import numpy as np

from configs.g1 import G1_CONFIG
from normalization import (FeatureStats, Normalization, fit_normalization,
                           fit_replay_normalization, unit_quaternion)
from replay_buffer import ReplayBuffer, ReplaySample
from robot_config import RobotConfig
from transitions import Transition
from world_model import WorldModel, WorldModelConfig
from world_model_training import WorldModelTrainer


ROBOT = RobotConfig(state_shapes=(("x", (2,)),), context_shapes=(("c", ()),),
                    action_dim=1, observation_dim=2, history_length=2)


def records(state, following, action=None, context=None):
    batch = state.shape[:-1]
    action = jnp.zeros(batch + (1,)) if action is None else action
    context = jnp.ones(batch + (1,)) if context is None else context
    return Transition(state, action, following, jnp.zeros(batch), jnp.zeros(batch, bool),
                      jnp.zeros(batch, bool), context, jnp.zeros_like(action),
                      state, following, {})


class TrainingTests(unittest.TestCase):
    def test_statistics_exclude_masked_and_nonfinite_rows(self):
        state = jnp.array([[[1., 10.], [3., 20.], [999., 999.], [jnp.nan, jnp.nan]]])
        data = records(state, state + 1., context=jnp.array([[[2.], [4.], [99.], [99.]]]))
        stats = fit_normalization(data, ROBOT, jnp.array([[True, True, False, True]]))
        self.assertEqual(int(stats.count), 2)
        np.testing.assert_allclose(stats.state.mean, [2.5, 15.5])
        np.testing.assert_allclose(stats.state.scale, np.sqrt([1.25, 25.25]), rtol=1e-6)
        np.testing.assert_allclose(stats.delta.mean, [1., 1.])
        np.testing.assert_allclose(stats.delta.scale, [.001, .001])
        np.testing.assert_allclose(stats.context.mean, [3.])
        np.testing.assert_allclose(stats.context.scale, [1.])

    def test_statistics_use_only_filled_replay_rows(self):
        replay = ReplayBuffer(ROBOT, capacity=5, sequence_length=1)
        data = records(jnp.ones((2, 2)) * 5, jnp.ones((2, 2)) * 7)
        storage = replay.add(replay.init(data), data)
        stats = jax.jit(lambda s: fit_replay_normalization(s, ROBOT))(storage)
        self.assertEqual(int(stats.count), 2)
        np.testing.assert_allclose(stats.state.mean, [6., 6.])
        for _ in range(6):
            storage = replay.add(storage, data)
        self.assertEqual(int(fit_replay_normalization(storage, ROBOT).count), 10)

    def test_quaternion_sign_unit_length_and_finite_zero_gradient(self):
        state = jnp.zeros((1, 2, 71)).at[..., 3].set(2.)
        following = state.at[..., 3].set(-3.)
        batch = (1, 2)
        data = Transition(state, jnp.zeros(batch + (29,)), following,
                          jnp.zeros(batch), jnp.zeros(batch, bool), jnp.zeros(batch, bool),
                          jnp.zeros(batch + (33,)), jnp.zeros(batch + (29,)),
                          jnp.zeros(batch + (93,)), jnp.zeros(batch + (93,)), {})
        stats = fit_normalization(data, G1_CONFIG)
        np.testing.assert_array_equal(stats.state.mean[3:7], 0.)
        np.testing.assert_array_equal(stats.state.scale[3:7], 1.)
        np.testing.assert_array_equal(stats.delta.mean[3:7], 0.)
        model = WorldModel(G1_CONFIG, WorldModelConfig(hidden_dims=(8,)))
        args = (data.state, data.action, data.context, stats)
        variables = model.init(jax.random.key(0), *args)
        prediction = model.apply(variables, *args)
        opposite = model.apply(variables, data.state.at[..., 3:7].multiply(-1.),
                               data.action, data.context, stats)
        np.testing.assert_allclose(prediction, opposite)
        np.testing.assert_allclose(jnp.linalg.norm(prediction[..., 3:7], axis=-1), 1., atol=1e-6)
        self.assertTrue(bool(jnp.isfinite(jax.jacfwd(unit_quaternion)(jnp.zeros(4))).all()))

    def test_rollout_uses_predictions_instead_of_later_recorded_states(self):
        model = WorldModel(ROBOT, WorldModelConfig(hidden_dims=()))
        trainer = WorldModelTrainer(model)
        stats = Normalization(FeatureStats(jnp.zeros(2), jnp.ones(2)),
                              FeatureStats(jnp.zeros(1), jnp.ones(1)),
                              FeatureStats(jnp.zeros(2), jnp.ones(2)), jnp.array(1))
        train = trainer.init(jax.random.key(0), stats)
        params = unfreeze(train.params)
        params["dynamics"]["output"]["kernel"] = jnp.zeros((4, 2)).at[0, 0].set(1.)
        params["dynamics"]["output"]["bias"] = jnp.zeros(2)
        state = jnp.full((2, 8, 2), 999.).at[:, 0].set(jnp.array([1., 0.]))
        target = jnp.zeros((2, 8, 2)).at[..., 0].set(2. ** jnp.arange(1, 9))
        data = records(state, target)
        predicted = trainer.rollout(params, stats, data)
        np.testing.assert_allclose(predicted, target)
        loss, per_step = trainer.loss(params, stats, data)
        np.testing.assert_allclose(loss, 0.)
        self.assertEqual(per_step.shape, (8,))

    def test_updates_reduce_loss_and_keep_statistics_fixed(self):
        action = jax.random.uniform(jax.random.key(2), (16, 8, 1), minval=-1., maxval=1.)
        context = jnp.full((16, 8, 1), .5)
        delta = jnp.repeat(.1 * action + .02 * context, 2, axis=-1)
        following = jnp.cumsum(delta, axis=1)
        state = jnp.concatenate((jnp.zeros((16, 1, 2)), following[:, :-1]), axis=1)
        data = records(state, following, action, context)
        stats = fit_normalization(data, ROBOT)
        trainer = WorldModelTrainer(WorldModel(ROBOT, WorldModelConfig(hidden_dims=(16, 16))),
                                    learning_rate=1e-3)
        train = trainer.init(jax.random.key(3), stats)
        sample = ReplaySample(data, jnp.zeros((16, 2, 2)), jnp.array(True), jnp.array(16))
        initial_loss = float(trainer.loss(train.params, stats, data)[0])
        update = jax.jit(trainer.update)
        for _ in range(100):
            train, metrics = update(train, sample)
        final_loss = float(trainer.loss(train.params, stats, data)[0])
        self.assertLess(final_loss, initial_loss * .3)
        self.assertEqual(int(train.step), 100)
        self.assertTrue(bool(metrics["updated"]))
        for actual, expected in zip(jax.tree.leaves(train.normalization), jax.tree.leaves(stats)):
            np.testing.assert_array_equal(actual, expected)
        skipped, metrics = update(train, sample._replace(valid=jnp.array(False)))
        self.assertFalse(bool(metrics["updated"]))
        for actual, expected in zip(jax.tree.leaves(skipped), jax.tree.leaves(train)):
            np.testing.assert_array_equal(actual, expected)
        bad_data = data._replace(next_state=data.next_state.at[0, 0, 0].set(jnp.nan))
        skipped, metrics = update(train, sample._replace(transitions=bad_data))
        self.assertFalse(bool(metrics["updated"]))
        self.assertEqual(int(skipped.step), int(train.step))

    def test_empty_warmup_is_rejected(self):
        data = records(jnp.zeros((1, 8, 2)), jnp.zeros((1, 8, 2)))
        stats = fit_normalization(data, ROBOT, jnp.zeros((1, 8), bool))
        trainer = WorldModelTrainer(WorldModel(ROBOT, WorldModelConfig(hidden_dims=(8,))))
        with self.assertRaises(ValueError):
            trainer.init(jax.random.key(0), stats)


if __name__ == "__main__":
    unittest.main()
