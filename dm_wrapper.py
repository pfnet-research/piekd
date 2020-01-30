from gym import core, spaces
from dm_control import suite
import dm_env
from dm_env import specs
from gym.utils import seeding
import gym
import numpy as np
import sys
import matplotlib.pyplot as plt

import pyglet
import numpy as np

from gym.envs.registration import register
import gym
import hashlib
from dm_control import suite

def make(domain_name, task_name, task_kwargs=None, visualize_reward=False):
    # register environment
    prehash_id = domain_name + task_name + str(task_kwargs) + str(visualize_reward)
    h = hashlib.md5(prehash_id.encode())
    gym_id = h.hexdigest()+'-v0'

    # avoid re-registering
    if gym_id not in gym_id_list:
        register(
            id=gym_id,
            entry_point='dm_control2gym.wrapper:DmControlWrapper',
            kwargs={'domain_name': domain_name, 'task_name': task_name, 'task_kwargs': task_kwargs,
                    'visualize_reward': visualize_reward, 'render_mode_list': render_mode_list}
        )
    # add to gym id list
    gym_id_list.append(gym_id)

    # make the Open AI env
    return gym.make(gym_id)

def create_render_mode(name, show=True, return_pixel=False, height=240, width=320, camera_id=-1, overlays=(),
             depth=False, scene_option=None):

    render_kwargs = { 'height': height, 'width': width, 'camera_id': camera_id,
                              'overlays': overlays, 'depth': depth, 'scene_option': scene_option}
    render_mode_list[name] = {'show': show, 'return_pixel': return_pixel, 'render_kwargs': render_kwargs}



# add procedurally generated environments
'''
@suite.swimmer.SUITE.add('proc')
def swimmer_k(**kwargs):
    return suite.swimmer.swimmer(**kwargs)

@suite.stacker.SUITE.add('proc')
def stack_k(k=2, observable=True, time_limit=suite.stacker._TIME_LIMIT, random=None):
    n_boxes = max(1, min(k, 4))
    if n_boxes != k:
        print('Input out of bounds. k set to: ',n_boxes)
    physics = suite.stacker.Physics.from_xml_string(*suite.stacker.make_model(n_boxes=n_boxes))
    task = suite.stacker.Stack(n_boxes, observable, random=random)
    return suite.control.Environment(
      physics, task, control_timestep=suite.stacker._CONTROL_TIMESTEP, time_limit=time_limit)

@suite.lqr.SUITE.add('proc')
def lqr_n_m(n=2, m=1, time_limit=suite.lqr._DEFAULT_TIME_LIMIT, random=None):
    _m = min(n, max(m,1))
    if _m != m:
        print('Input error. m should be <=n. m set to: ', _m)
    return suite.lqr._make_lqr(n_bodies=n,
                   n_actuators=_m,
                   control_cost_coef=suite.lqr._CONTROL_COST_COEF,
                   time_limit=time_limit,
                   random=random)

@suite.cartpole.SUITE.add('proc')
def k_poles(k=2, swing_up=True, sparse=False, time_limit=suite.cartpole._DEFAULT_TIME_LIMIT, random=None):
    physics = suite.cartpole.Physics.from_xml_string(*suite.cartpole.get_model_and_assets(num_poles=k))
    task = suite.cartpole.Balance(swing_up=swing_up, sparse=sparse, random=random)
    return suite.control.Environment(physics, task, time_limit=time_limit)

'''

gym_id_list = []
render_mode_list = {}
create_render_mode('human', show=True, return_pixel=False)
create_render_mode('rgb_array', show=False, return_pixel=True)
create_render_mode('human_rgb_array', show=True, return_pixel=True)

class DmControlViewer:
    def __init__(self, width, height, depth=False):
        self.window = pyglet.window.Window(width=width, height=height, display=None)
        self.width = width
        self.height = height

        self.depth = depth

        if depth:
            self.format = 'RGB'
            self.pitch = self.width * -3
        else:
            self.format = 'RGB'
            self.pitch = self.width * -3

    def update(self, pixel):
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        if self.depth:
            pixel = np.dstack([pixel.astype(np.uint8)] * 3)
        pyglet.image.ImageData(self.width, self.height, self.format, pixel.tobytes(), pitch=self.pitch).blit(0, 0)
        self.window.flip()

    def close(self):
        self.window.close()


class DmcDiscrete(gym.spaces.Discrete):
    def __init__(self, _minimum, _maximum):
        super().__init__(_maximum - _minimum)
        self.offset = _minimum

def convertSpec2Space(spec, clip_inf=False):
    if spec.dtype == np.int:
        # Discrete
        return DmcDiscrete(spec.minimum, spec.maximum)
    else:
        # Box
        if type(spec) is specs.Array:
            return spaces.Box(-np.inf, np.inf, shape=spec.shape)
        elif type(spec) is specs.BoundedArray:
            _min = spec.minimum
            _max = spec.maximum
            if clip_inf:
                _min = np.clip(spec.minimum, -sys.float_info.max, sys.float_info.max)
                _max = np.clip(spec.maximum, -sys.float_info.max, sys.float_info.max)

            if np.isscalar(_min) and np.isscalar(_max):
                # same min and max for every element
                return spaces.Box(_min, _max, shape=spec.shape)
            else:
                # different min and max for every element
                return spaces.Box(_min + np.zeros(spec.shape),
                                  _max + np.zeros(spec.shape))
        else:
            raise ValueError('Unknown spec!')

def convertOrderedDict2Space(odict):
    if len(odict.keys()) == 1:
        # no concatenation
        return convertSpec2Space(list(odict.values())[0])
    else:
        # concatentation
        numdim = sum([np.int(np.prod(odict[key].shape)) for key in odict])
        return spaces.Box(-np.inf, np.inf, shape=(numdim,))


def convertObservation(spec_obs):
    if len(spec_obs.keys()) == 1:
        # no concatenation
        return list(spec_obs.values())[0]
    else:
        # concatentation
        numdim = sum([np.int(np.prod(spec_obs[key].shape)) for key in spec_obs])
        space_obs = np.zeros((numdim,))
        i = 0
        for key in spec_obs:
            space_obs[i:i+np.prod(spec_obs[key].shape)] = spec_obs[key].ravel()
            i += np.prod(spec_obs[key].shape)
        return space_obs

class DmControlWrapper(core.Env):

    def __init__(self, domain_name, task_name, task_kwargs=None, visualize_reward=False, render_mode_list=None):

        self.dmcenv = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs=task_kwargs,
                                 visualize_reward=visualize_reward)

        # convert spec to space
        self.action_space = convertSpec2Space(self.dmcenv.action_spec(), clip_inf=True)
        self.observation_space = convertOrderedDict2Space(self.dmcenv.observation_spec())

        if render_mode_list is not None:
            self.metadata['render.modes'] = list(render_mode_list.keys())
            self.viewer = {key:None for key in render_mode_list.keys()}
        else:
            self.metadata['render.modes'] = []

        self.render_mode_list = render_mode_list

        # set seed
        self._seed()

    def getObservation(self):
        return convertObservation(self.timestep.observation)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.timestep = self.dmcenv.reset()
        return self.getObservation()

    def _step(self, a):

        if type(self.action_space) == DmcDiscrete:
            a += self.action_space.offset
        self.timestep = self.dmcenv.step(a)

        return self.getObservation(), self.timestep.reward, self.timestep.last(), {}


    def _render(self, mode='human', close=False):

        self.pixels = self.dmcenv.physics.render(**self.render_mode_list[mode]['render_kwargs'])
        if close:
            if self.viewer[mode] is not None:
                self._get_viewer(mode).close()
                self.viewer[mode] = None
            return
        elif self.render_mode_list[mode]['show']:
            self._get_viewer(mode).update(self.pixels)



        if self.render_mode_list[mode]['return_pixel']:

            return self.pixels

    def _get_viewer(self, mode):
        if self.viewer[mode] is None:
            self.viewer[mode] = DmControlViewer(self.pixels.shape[1], self.pixels.shape[0], self.render_mode_list[mode]['render_kwargs']['depth'])
        return self.viewer[mode]

