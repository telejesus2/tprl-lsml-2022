import time
from environments.gym import VideoRecorderWrapper


class Agent(object):
    def __init__(
        self,
        env,
    ):
        self.env = env
        self.eval_env = VideoRecorderWrapper(env.duplicate(1)[0])

    def update(self):
        pass

    def act(self, ob, eval=False):
        pass

    def log_progress(self):
        pass

    def collect_rollouts(self, itr, render=False):
        pass

    def eval(self, num_episodes=1, render=False):
        returns = []
        lengths = []
        for _ in range(num_episodes):
            episode_rewards = self._run_eval_episode(render)
            returns.append(sum(episode_rewards))
            lengths.append(len(episode_rewards))
        if render:
            self.eval_env.close()
        return returns, lengths

    def _run_eval_episode(self, render):
        ob = self.eval_env.reset(record=render)
        returns = []
        done = False
        while not done:
            if render:
                self.eval_env.render()
                # time.sleep(0.1)
            ac = self.act(ob, eval=True)
            ob, rew, done, _ = self.eval_env.step(ac)
            returns.append(rew)
        return returns
