"""Unity environment wrapper

Modify from: atari_wrapper in openai/baselines
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""
from collections import deque

import gym
from gym_unity.envs.unity_env import UnityEnv
import numpy as np


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2, ) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, chw=False):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.chw = chw
        shp = env.observation_space.shape
        if chw:
            # Can only use with grayscale e.g. turn to (4, 84, 84)
            assert shp[0] == 1, 'Can only use with grayscale'
            shape = ((shp[0] * k, ) + shp[1:])
        else:
            shape = shp[:-1] + (shp[-1] * k, )

        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=shape,
                                                dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames), self.chw)


class LazyFrames(object):
    def __init__(self, frames, chw=False):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        # self._out = None  # cache the ndarray, that will increase the memory usage
        self.chw = chw

    def _force(self):
        # Do not use cache self._out
        out = None
        if self._frames is not None:
            if self.chw:  # (4, 84, 84)
                out = np.concatenate(self._frames, axis=0)
            else:  # (84, 84, 4)
                out = np.concatenate(self._frames, axis=-1)
            # self._frames = None
        return out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


class CHWStyle(gym.ObservationWrapper):
    def __init__(self, env):
        """Convert HWC (height x width x channel) to CHW
        """

        super().__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=(shape[-1], shape[0], shape[1]),
                                                dtype=env.observation_space.dtype)

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))


def wrap_unity_env(env_path, frame_skip=0, frame_stack=False, chw_style=False, **unity_config):

    worker_id = unity_config.get('port', 9527)
    use_visual = unity_config.get('use_visual', True)
    uint8_visual = unity_config.get('uint8_visual', True)
    flatten_branched = unity_config.get('flatten_branched', True)

    env = UnityEnv(env_path,
                   worker_id=worker_id,
                   use_visual=use_visual,
                   uint8_visual=uint8_visual,
                   flatten_branched=flatten_branched)

    if frame_skip > 0:
        env = MaxAndSkipEnv(env, frame_skip)
    if chw_style:
        env = CHWStyle(env)
    if frame_stack:
        env = FrameStack(env, frame_stack, chw_style)

    return env
