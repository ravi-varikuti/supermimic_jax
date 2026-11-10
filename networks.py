"""Reusable MLP building block for dynamics and future policy networks."""

from flax import linen as nn


class MLP(nn.Module):
    hidden_dims: tuple[int, ...]
    output_dim: int
    activation: str = "elu"
    output_kernel_scale: float = 1.0
    layer_norm: bool = False
    zero_output: bool = False

    @nn.compact
    def __call__(self, inputs):
        activations = {"elu": nn.elu, "relu": nn.relu, "silu": nn.silu, "tanh": nn.tanh}
        if self.activation not in activations:
            raise ValueError(f"Unknown activation: {self.activation}")
        activate = activations[self.activation]
        x = inputs
        for i, width in enumerate(self.hidden_dims):
            x = nn.Dense(width, name=f"hidden_{i}")(x)
            if self.layer_norm:
                x = nn.LayerNorm(name=f"norm_{i}")(x)
            x = activate(x)
        return nn.Dense(
            self.output_dim, name="output",
            kernel_init=nn.initializers.zeros if self.zero_output else nn.initializers.variance_scaling(
                self.output_kernel_scale ** 2, "fan_in", "truncated_normal",
            ),
        )(x)
