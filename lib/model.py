import torch
from activation import SineActivation


# Define a simple neural network model
# It seems like GLSL get's maxed out anything 32x32 or greater
# 16x16 seems to work pretty well with 4 layers...
# Sometimes I've tried 6 but it will probably be too much for smaller
# webgl hardware (like mobile)
#
# activation choices:
#   "relu"  — original behaviour, piecewise linear, bad at high frequencies
#   "sine"  — SIREN: sin(omega_0 * x), excellent at high frequencies,
#             requires siren_init() to be called after construction
#   "tanh"  — smooth, better than relu for INRs but slower than sine
class Neuralistica(torch.nn.Module):
    def __init__(self, input_size=16, output_size=3, hidden_size=16, hidden_count=4, activation="sine"):
        super().__init__()
        self.activation = activation
        self.fullness(input_size, output_size, hidden_size, hidden_count)
        if activation == "sine":
            self.siren_init()

    def _make_activation(self, is_first=False):
        if self.activation == "sine":
            return SineActivation(omega_0=30.0 if is_first else 1.0)
        elif self.activation == "tanh":
            return torch.nn.Tanh()
        else:  # relu — original behaviour
            return torch.nn.ReLU()

    def fullness(self, input_size, output_size, hidden_size, hidden_count):
        layers = []
        layers.append(torch.nn.Linear(input_size, hidden_size))
        layers.append(self._make_activation(is_first=True))
        for _ in range(hidden_count):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(self._make_activation(is_first=False))
        layers.append(torch.nn.Linear(hidden_size, output_size))
        # No sigmoid on output — sine networks produce outputs in [-1,1]
        # naturally. We clamp to [0,1] after. relu/tanh keep sigmoid for compat.
        if self.activation != "sine":
            layers.append(torch.nn.Sigmoid())
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def siren_init(self, omega_0=30.0):
        """
        SIREN weight initialisation from Sitzmann et al. 2020.
        Without this the sine network trains very poorly — the init scheme is
        what makes SIREN actually work. Called automatically from __init__
        when activation="sine".

        First linear layer: uniform in [-1/fan_in, 1/fan_in]
        Hidden linear layers: uniform in [-sqrt(6/fan_in)/omega_0, +sqrt(6/fan_in)/omega_0]

        This preserves the distribution of activations across layers so that
        the network can represent a wide range of frequencies from the start.
        """
        linear_idx = 0
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                n = layer.weight.shape[1]  # fan_in
                if linear_idx == 0:
                    # First layer — wider init so it spans the full input range
                    torch.nn.init.uniform_(layer.weight, -1.0 / n, 1.0 / n)
                else:
                    bound = (6.0 / n) ** 0.5 / omega_0
                    torch.nn.init.uniform_(layer.weight, -bound, bound)
                linear_idx += 1
