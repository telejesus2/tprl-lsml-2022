import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal
from networks.base import Function

POLICY = ['gaussian', 'squashed-gaussian', 'softmax', 'deterministic']


class PolicyNetwork(nn.Module):
    def __init__(self, ob_dim, ac_dim, hidden):
        super().__init__()
        # hidden layers
        self.fc = nn.ModuleList()
        nodes = [ob_dim] + hidden
        for i in range(len(hidden)):
            self.fc.append(nn.Linear(nodes[i], nodes[i+1]))
        # output layer
        self.out = nn.Linear(nodes[-1], ac_dim)

    def forward(self, x):
        for fc in self.fc:
            x = F.relu(fc(x))
        x = self.out(x)
        return x


class SquashedPolicyNetwork(PolicyNetwork):
    def __init__(self, ob_dim, ac_dim, max_ac, min_ac, hidden):
        super().__init__(ob_dim, ac_dim, hidden)
        self._center = torch.tensor(max_ac + min_ac) / 2
        self._scale = torch.tensor(max_ac - min_ac) / 2

    def _squash(self, x):
        return torch.tanh(x) * self._scale + self._center

    def to(self, device):
        self._center = self._center.to(device)
        self._scale = self._scale.to(device)
        return super().to(device)


class SoftmaxPolicyNetwork(PolicyNetwork):
    def forward(self, x):
        """
        :return: full distribution if training, deterministic action otherwise
        """
        out = super().forward(x)
        probs = F.softmax(out, dim=-1)
        if not self.training:
            return torch.argmax(probs, dim=-1)
        else:
            return Categorical(probs)


class GaussianPolicyNetwork(PolicyNetwork):
    def __init__(self, ob_dim, ac_dim, hidden):
        super().__init__(ob_dim, ac_dim, hidden)
        log_std = -0.5 * np.ones(ac_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, x):
        """
        :return: full distribution if training, deterministic action otherwise
        """
        mu = super().forward(x)
        if not self.training:
            return mu
        else:
            std = torch.exp(self.log_std)
            return Independent(Normal(mu, std), 1)  # otherwise needs normal.log_prob(ac).sum(axis=-1)


class SquashedGaussianPolicyNetwork(SquashedPolicyNetwork):
    def __init__(self, ob_dim, ac_dim, max_ac, min_ac, hidden):
        super().__init__(ob_dim, 2 * ac_dim, max_ac, min_ac, hidden)

    def forward(self, x):
        """
        :return: full distribution if training, deterministic action otherwise
        """
        out = super().forward(x)
        ac_dim = out.shape[-1] // 2
        mu = self._squash(out[:, :ac_dim])
        if not self.training:
            return mu
        else:
            log_std = out[:, ac_dim:] * self._scale
            std = torch.exp(log_std)
            return Independent(Normal(mu, std), 1)


class DeterministicPolicyNetwork(SquashedPolicyNetwork):
    def forward(self, x):
        """
        :return: deterministic action
        """
        out = super().forward(x)
        return self._squash(out)


class Policy(Function):
    def __init__(self, ob_dim, ac_dim, policy, max_ac=None, min_ac=None, hidden=[64, 64], lr=0.0005, target=False):
        
        if policy == 'softmax':
            net = SoftmaxPolicyNetwork(ob_dim, ac_dim, hidden)
        elif policy == 'gaussian':
            net = GaussianPolicyNetwork(ob_dim, ac_dim, hidden)
        elif policy == 'squashed-gaussian':
            net = SquashedGaussianPolicyNetwork(ob_dim, ac_dim, max_ac, min_ac, hidden)
        elif policy == 'deterministic':
            net = DeterministicPolicyNetwork(ob_dim, ac_dim, max_ac, min_ac, hidden)
        else:
            raise KeyError('Unknown policy name!')

        opt = optim.Adam(net.parameters(), lr=lr)

        super().__init__(net, opt, target)
