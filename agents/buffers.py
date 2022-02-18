import numpy as np
import scipy.signal
import torch


class ReplayBuffer(object):
    """ Expects tuples of (state, next_state, action, reward, done) """

    def __init__(self, device, max_size=1e6):
        self.device = device
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def can_sample(self, batch_size):
        return batch_size <= len(self.storage)

    def sample(self, batch_size):
        # keys = np.random.choice(len(self.storage), size=batch_size, replace=False)
        keys = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[key] for key in keys]

        x, y, u, r, d = [], [], [], [], []
        for (X, Y, U, R, D) in batch:
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
        return {
            'states': torch.FloatTensor(np.array(x)).to(self.device),
            'next_states': torch.FloatTensor(np.array(y)).to(self.device),
            'actions': torch.FloatTensor(np.array(u)).to(self.device),
            'rewards': torch.FloatTensor(np.array(r).reshape(-1, 1)).to(self.device),
            'done_mask': torch.FloatTensor(np.array(d).reshape(-1, 1)).to(self.device),
        }


class OnPolicyBuffer(object):
    def __init__(self, device, gae_lam=None):
        self.device = device
        self.gae_lam = gae_lam
        self.flush()
        self.obs_ep, self.acs_ep, self.rews_ep = [], [], []

    def add_ob(self, ob):
        self.obs_ep.append(ob)

    def add(self, ob, ac, rew):
        self.obs_ep.append(ob)
        self.acs_ep.append(ac)
        self.rews_ep.append(rew)

    def flush_episode(self, done):
        self.obs.append(np.array(self.obs_ep[:-1], dtype=np.float32, copy=False))
        self.acs.append(np.array(self.acs_ep, dtype=np.float32, copy=False))
        self.rews.append(np.array(self.rews_ep, dtype=np.float32, copy=False))
        self.last_obs.append(np.array(self.obs_ep[-1], dtype=np.float32, copy=False))
        self.last_dones.append(done)
        self.ep_t.append(self.ep_t[-1] + len(self.rews_ep))

        self.obs_ep, self.acs_ep, self.rews_ep = [], [], []

    def flush(self):
        self.obs, self.acs, self.rews  = [], [], []
        self.last_obs, self.last_dones, self.last_values = [], [], []
        self.ep_t = [0]

    def observations(self):
        return torch.FloatTensor(np.concatenate(self.obs)).to(self.device)

    def next_observations(self):
        return torch.FloatTensor(np.concatenate([np.concatenate((obs[1:], [last_ob]))
            for obs, last_ob in zip(self.obs, self.last_obs)])).to(self.device)

    def actions(self):
        return torch.FloatTensor(np.concatenate(self.acs)).to(self.device)

    def rewards(self):
        return torch.FloatTensor(np.concatenate(self.rews)).to(self.device)

    def terminals(self):
        dones = np.zeros(self.ep_t[-1])
        for i in range(len(self.ep_t) - 1):
            dones[self.ep_t[i + 1] - 1] = self.last_dones[i]
        return torch.FloatTensor(dones).to(self.device)

    def returns(self, gamma):
        rets = []
        for rews in self.rews:
            rets.extend([sum(r * gamma ** i for i, r in enumerate(rews))] * len(rews))
        return torch.FloatTensor(np.array(rets, dtype=np.float32)).to(self.device)

    def returns_to_go(self, gamma, add_last_values=False, update_last_values_with=None):
        rets = []
        if add_last_values:
            if update_last_values_with is not None:
                self._compute_last_values(update_last_values_with)
            for i, rews in enumerate(self.rews):
                full_rews = np.append(self.rews[i], self.last_values[i])
                rets.extend(self._returns_to_go(full_rews, gamma)[:-1])
        else:
            for rews in self.rews:
                rets.extend(self._returns_to_go(rews, gamma))
        return torch.FloatTensor(np.array(rets, dtype=np.float32)).to(self.device)
        
    def _returns_to_go(self, rews, gamma):
        return scipy.signal.lfilter([1], [1, float(- gamma)], rews[::-1], axis=0)[::-1]

    @torch.no_grad()
    def _compute_last_values(self, net):
        if 0 in self.last_dones:
            self.last_values = (1 - np.array(self.last_dones)) * net(
                torch.FloatTensor(np.array(self.last_obs)).to(self.device)).view(-1).cpu().numpy()
        else:
            self.last_values = np.zeros(len(self.last_dones))
