import numpy as np
import gym

class NoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, scale=0.1):
        super(NoiseWrapper, self).__init__(env)
        self.scale = scale

    def observation(self, observation):
        dtype = observation.dtype
        noise = np.random.normal(np.zeros_like(observation), self.scale).astype(dtype)
        return noise + observation


