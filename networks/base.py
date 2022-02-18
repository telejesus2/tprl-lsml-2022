import torch
import copy


class Function(object):
    def __init__(self, network, optimizer, target):
        self.net = network
        self.opt = optimizer

        self.target = target
        if target:
            self.target_net = self._init_target(self.net)

    def _init_target(self, net):
        target_net = copy.deepcopy(net)
        for p in target_net.parameters():
            p.requires_grad = False
        return target_net

    def forward(self, input, eval=False):
        istrain = self.net.training
        tmp = False if eval is True else istrain
        self.net.train(tmp)
        if isinstance(input, tuple):
            output = self.net(*input)
        else:
            output = self.net(input)
        self.net.train(istrain)
        return output

    def optimize(self, loss, grad_norm_clip_val=None, retain_graph=False):
        self.opt.zero_grad()
        # if grad_norm_clip_val is not None:
        #     for param in self.net.parameters():
        #         param.register_hook(lambda grad: grad.clamp_(-grad_norm_clip_val, grad_norm_clip_val)) 
        loss.backward(retain_graph=retain_graph)
        if grad_norm_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_norm_clip_val)
        self.opt.step()

    def to_(self, device):
        self.net.to(device)
        if self.target:
            self.target_net.to(device)

    def sync_target(self, tau=None):
        if tau is None:
            self.target_net.load_state_dict(self.net.state_dict())
        else:
            with torch.no_grad():
                for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, name):
        torch.save(self.net.state_dict(), name + '.pt')
        if self.target:
            torch.save(self.target_net.state_dict(), name + '_target.pt')
