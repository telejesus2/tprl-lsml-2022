import time
import numpy as np
from agents.common_all import Agent
from agents.buffers import OnPolicyBuffer


class OnPolicyAgent(Agent):
    def __init__(
            self,
            env,
            device,
            batch_size,
            max_path_frames,
            gae_lam=0.97,
            render_interval=10,
    ):
        super(OnPolicyAgent, self).__init__(
            env,
        )
        self.device = device
        self.render_interval = render_interval
        self.batch_size = batch_size
        self.max_path_frames = min(max_path_frames, env.max_episode_steps - 1)

        # buffer
        self.buffer = OnPolicyBuffer(device)      

        # logging
        self.on_policy = True
        self._episode = 0
        self._frame = 0
        self._start = time.time()
        self._itr_start_time = self._start
        self._frames_last_iter = 0

    def log_progress(self):
        returns = [rews.sum() for rews in self.buffer.rews]
        ep_lengths = [len(rews) for rews in self.buffer.rews]
        stats = {
            'Time': (time.time() - self._start) / 60.,
            'Timesteps': self._frame,
            'TimestepsThisBatch': self._frames_last_iter,
            'fps': (self._frames_last_iter / (time.time() - self._itr_start_time)),
            'MeanReturn': np.mean(returns),
            'StdReturn': np.std(returns),
            'MaxReturn': np.max(returns),
            'MinReturn': np.min(returns),
            'EpLenMean': np.mean(ep_lengths),
            'EpLenStd': np.std(ep_lengths),
            'Episodes': self._episode,  
        }
        self._itr_start_time = time.time()
        return stats

    def collect_rollouts(self, itr, render=False):
        self.buffer.flush()
        frames_this_iter = 0
        while True:
            render = (frames_this_iter == 0 and (itr % self.render_interval == 0) and render)
            ep_length = self._collect_episode(render)
            frames_this_iter += ep_length
            if frames_this_iter > self.batch_size - 1:
                break
        self._frame += frames_this_iter
        self._frames_last_iter = frames_this_iter

    def _collect_episode(self, render):
        ob = self.env.reset()
        self.buffer.add_ob(ob)
        episode_frames = 0
        while True:
            if render:
                self.env.render()
                # time.sleep(0.1)
            ac = self.act(ob)
            ob, rew, done, _ = self.env.step(ac)
            self.buffer.add(ob, ac, rew)
            episode_frames += 1
            if done or episode_frames > self.max_path_frames - 1:
                self.buffer.flush_episode(done)
                break
        if render:
            self.env.close()
        self._episode += 1
        return episode_frames
