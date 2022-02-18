import gym
import pybullet_envs
# from gym.wrappers.monitoring import video_recorder
from environments import video_recorder
import torch
import numpy as np

discrete_envs = {
    'car':      'MountainCar-v0',
    'lunar':    'LunarLander-v2',
    'cartpole': 'CartPole-v0'
}
continuous_envs = {
    'pendulum':         'Pendulum-v0',
    'car-continuous':   'MountainCarContinuous-v0',
    'lunar-continuous': 'LunarLanderContinuous-v2',
    'cheetah':          'HalfCheetahBulletEnv-v0',
    'hopper':           'HopperBulletEnv-v0',
    'walker':           'Walker2DBulletEnv-v0',
    'ant':              'AntBulletEnv-v0',
    'reacher':          'ReacherBulletEnv-v0',
    'inverted-pendulum':         'InvertedPendulumBulletEnv-v0',
    'double-pendulum':  'InvertedDoublePendulumBulletEnv-v0',
}
bullet_envs = {
    'cheetah',
    'hopper',
    'walker',
    'ant',
    'reacher',
    'inverted-pendulum',
    'doublependulum',
}


class GymEnv(object):
    def __init__(self, env_name, device):
        '''
        :param env_name: short name of the gym env (e.g. 'car' for 'MountainCar-v0')
        '''
        self.device = device
        self._short_name = env_name
        self.is_bullet = env_name in bullet_envs
        if env_name in discrete_envs.keys():
            self.name = discrete_envs[env_name]
            self.is_discrete = True
        elif env_name in continuous_envs.keys():
            self.name = continuous_envs[env_name]
            self.is_discrete = False
        else:
            raise KeyError('env_name not valid!')

        self._env = gym.make(self.name)
        self.metadata = self._env.metadata
        self.max_action = self.action_space.high if not self.is_discrete else None
        self.min_action = self.action_space.low if not self.is_discrete else None
        self.action_dim = self.action_space.shape[0] if not self.is_discrete else self.action_space.n
        self.observation_dim = self.state_space.shape[0]
        self.max_episode_steps = self._env.spec.max_episode_steps
        self.reward_range = self._env.reward_range

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def state_space(self):
        return self._env.observation_space

    @property
    def unwrapped(self):
        return self

    def reset(self):
        return self._env.reset()

    def step(self, ac):
        return self._env.step(ac)

    def render(self, mode='human', **kwargs):
        return self._env.render(mode=mode, **kwargs)

    def close(self):
        return self._env.close()

    def seed(self, seed):
        self._env.seed(seed)

    def process_state(self, ob):
        # to tensor of shape [1 , observation_dim]
        return torch.FloatTensor(ob).view(1, -1).to(self.device)

    def process_action(self, ac):
        # from tensor of shape [1]              if env is discrete
        # from tensor of shape [1, action_dim]  if env is continuous
        if self.is_discrete:
            return ac.item()
        else:
            return np.squeeze(ac.cpu().detach().numpy(), 0)

    def duplicate(self, n):
        return [GymEnv(self._short_name, self.device) for _ in range(n)]


class Wrapper(GymEnv):
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def state_space(self):
        return self.env.observation_space

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self):
        return self.env.reset()

    def step(self, ac):
        return self.env.step(ac)

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def process_state(self, ob):
        return self.env.process_state(ob)

    def process_action(self, ac):
        return self.env.process_action(ac)

    def duplicate(self, n):
        return self.env.duplicate(n)


class VideoRecorderWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.record = False
        self.recording = False
        self.video_name = './video'

    def setup_recording(self, name):
        self.video_name = name

    def reset(self, record=False):
        self.record = record
        return self.env.reset()

    def step(self, ac):
        if self.record:
            if not self.recording:
                self.recording = True
                self.video_recorder = video_recorder.VideoRecorder(
                    env=self.env,
                    base_path=self.video_name)
            self.video_recorder.capture_frame()
        return self.env.step(ac)

    def close(self):
        if self.recording:
            self.recording = False
            self.video_recorder.close()
        return self.env.close()
