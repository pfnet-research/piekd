import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class SparseHalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        if abs(xposafter - 5.0) > 0.1:
            reward = -1
        else:
            reward = 0
        done = False
        return ob, reward, done, dict(pos=xposafter)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
