class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        sizing = 16

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, 16),  torch.nn.ReLU(),
            # note: 10 seems to be about the limit for 16x16
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 3), torch.nn.ReLU(),
        )

