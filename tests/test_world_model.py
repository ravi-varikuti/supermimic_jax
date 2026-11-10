import unittest

from flax.core import unfreeze
import jax
import jax.numpy as jnp
import numpy as np

from configs.g1 import G1_CONFIG
from robot_config import RobotConfig
from world_model import WorldModel, WorldModelConfig


class WorldModelTests(unittest.TestCase):
    def test_g1_default_architecture(self):
        model = WorldModel(G1_CONFIG)
        inputs = (jnp.zeros((2, 71)), jnp.zeros((2, 29)), jnp.zeros((2, 33)))
        variables = model.init(jax.random.key(0), *inputs)
        self.assertEqual(sum(x.size for x in jax.tree.leaves(variables)), 8_812_615)
        result = jax.jit(model.apply)(variables, *inputs)
        self.assertEqual(result.shape, (2, 71))
        self.assertTrue(bool(jnp.isfinite(result).all()))

    def test_input_order_and_residual(self):
        robot = RobotConfig(state_shapes=(("x", (2,)),), context_shapes=(("friction", ()),),
                            action_dim=1, observation_dim=3)
        model = WorldModel(robot, WorldModelConfig(hidden_dims=()))
        inputs = (jnp.array([[4., 5.]]), jnp.array([[2.]]), jnp.array([[3.]]))
        variables = unfreeze(model.init(jax.random.key(0), *inputs))
        output = variables["params"]["dynamics"]["output"]
        output["kernel"] = jnp.zeros((4, 2)).at[2, 0].set(2.).at[3, 1].set(3.)
        output["bias"] = jnp.zeros(2)
        np.testing.assert_allclose(model.apply(variables, *inputs), [[8., 14.]])
        direct = WorldModel(robot, WorldModelConfig(hidden_dims=(), predict_delta=False))
        np.testing.assert_allclose(direct.apply(variables, *inputs), [[4., 9.]])

    def test_other_robot_time_axes_and_gradients(self):
        robot = RobotConfig(state_shapes=(("x", (5,)),), context_shapes=(),
                            action_dim=2, observation_dim=4)
        model = WorldModel(robot, WorldModelConfig(hidden_dims=(16, 16), activation="silu"))
        state = jnp.full((3, 8, 5), .2)
        action = jnp.full((3, 8, 2), .3)
        context = jnp.zeros((3, 8, 0))
        variables = model.init(jax.random.key(1), state, action, context)
        result = jax.jit(model.apply)(variables, state, action, context)
        self.assertEqual(result.shape, (3, 8, 5))
        gradients = jax.grad(lambda s, a: model.apply(variables, s, a, context).sum(),
                             argnums=(0, 1))(state, action)
        for grad in gradients:
            self.assertTrue(bool(jnp.isfinite(grad).all()))
            self.assertGreater(float(jnp.linalg.norm(grad)), 0.)

        target = state + .1
        loss = lambda p: jnp.mean((model.apply({"params": p}, state, action, context) - target) ** 2)
        params = variables["params"]
        initial_loss = float(loss(params))

        @jax.jit
        def update(params):
            gradients = jax.grad(loss)(params)
            return jax.tree.map(lambda p, g: p - .1 * g, params, gradients)

        for _ in range(20):
            params = update(params)
        self.assertLess(float(loss(params)), initial_loss * .5)

    def test_bad_shapes(self):
        model = WorldModel(G1_CONFIG, WorldModelConfig(hidden_dims=(8,)))
        with self.assertRaises(ValueError):
            model.init(jax.random.key(0), jnp.zeros((2, 71)), jnp.zeros(29), jnp.zeros((2, 33)))


if __name__ == "__main__":
    unittest.main()
