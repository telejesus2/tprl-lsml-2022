import time
import numpy as np
from agents.common_all import Agent
from agents.buffers import ReplayBuffer


class OffPolicyAgent(Agent):
    def __init__(
        self,
        env,
        device,
        batch_size,
        learning_starts,	# start learning after x frames
        learning_freq,		# update model every x frames
        replay_buffer_size,
        max_path_frames,
    ):
        super(OffPolicyAgent, self).__init__(
            env,
        )
        self.device = device
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.learning_freq = learning_freq
        self.max_path_frames = min(max_path_frames, env.max_episode_steps - 1)

        # replay buffer
        self.replay_buffer = ReplayBuffer(device, replay_buffer_size)

        # utilities for collect_rollouts()
        self._last_ob = self.env.reset()
        self._episode_frame = 0
        self._episode_rewards = []  # list of rewards at each timestep during an episode
        self._frame = 0

        # logging
        self.on_policy = False
        self._start = time.time()      
        self._episode_returns = []  # list of returns (sum of rewards) for each episode
        self._episode_lengths = []  # list of lengths for each episode
        self._best_mean_episode_reward = - np.inf

    def log_progress(self):
        mean_episode_reward = None
        if len(self._episode_returns) > 0:
            mean_episode_reward = np.mean(self._episode_returns[-100:])
            if len(self._episode_returns) > 100:
                self._best_mean_episode_reward = max(
                    self._best_mean_episode_reward, mean_episode_reward)
        stats = {
            'Time': (time.time() - self._start) / 60.,
            'Timesteps': self._frame,
            'MeanReturn': mean_episode_reward,
            'BestMeanReturn': self._best_mean_episode_reward,
            'Episodes': len(self._episode_returns),
            'Exploration': self.exploration.value(self._frame),
        }
        return stats

    def collect_rollouts(self, itr, render=False):
        frames_this_iter = 0
        while True:
            ac = self.act(self._last_ob)
            ob, rew, done, _ = self.env.step(ac)
            self.replay_buffer.add((self._last_ob, ob, ac, rew, done))
            self._last_ob = ob
            self._episode_rewards.append(rew)
            frames_this_iter += 1
            self._episode_frame += 1
            self._frame += 1

            if done or self._episode_frame > self.max_path_frames - 1:
                self._last_ob = self.env.reset()
                self._episode_returns.append(sum(self._episode_rewards))
                self._episode_lengths.append(len(self._episode_rewards))
                self._episode_rewards = []
                self._episode_frame = 0

            if self._frame > self.learning_starts - 1 and \
               frames_this_iter > self.learning_freq - 1 and \
               self.replay_buffer.can_sample(self.batch_size):
                break
