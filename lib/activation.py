import torch


# Sine activation for SIREN networks.
# omega_0 controls the frequency of the first layer — 30.0 is the value
# from the original SIREN paper (Sitzmann et al. 2020) and works well in
# practice. Hidden layers use omega_0=1.0 so they don't re-amplify.
# In GLSL this is just: sin(sum) — single instruction, basically free.
class SineActivation(torch.nn.Module):
    def __init__(self, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)
