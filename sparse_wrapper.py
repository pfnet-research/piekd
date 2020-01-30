import gym

class SparseRewardWrapper(gym.Wrapper):
    def __init__(self, env, sparse_level=-1, timestep_limit=-1):
        super(SparseRewardWrapper, self).__init__(env)
        self.sparse_level = sparse_level
        self.timestep_limit = timestep_limit
        self.acc_reward = 0
        self.acc_t = 0

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        self.acc_t += 1

        if self.timestep_limit > 0 and (self.acc_t) >= self.timestep_limit:
            done = True

        if self.sparse_level == 0:
            return obs, rew, done, info

        self.acc_reward += rew
        ret_rew = 0
        if self.sparse_level != -1:
            if done or (self.acc_t > 0 and self.acc_t % self.sparse_level == 0):
                ret_rew = self.acc_reward
                self.acc_reward = 0
        else:
            if done:
                ret_rew = self.acc_reward
                self.acc_reward = 0

        return obs, ret_rew, done, info

    def reset(self, **kwargs):
        self.acc_t = 0
        self.acc_reward = 0
        return self.env.reset(**kwargs)

