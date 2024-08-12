#https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py
"""Implementation of Atari 2600 Preprocessing following the guidelines of Machado et al., 2018."""
import numpy as np
import random

import gym
from gym.spaces import Box

try:
    import cv2
except ImportError:
    cv2 = None

class LifeLostWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(LifeLostWrapper, self).__init__(env)
        self.last_lives = 0

    def reset(self, **kwargs):
        self.last_lives = self.env.ale.lives()  # Initialize lives at reset
        return self.env.reset(**kwargs)

    def action(self, action):
        current_lives = self.env.ale.lives()
        
        if current_lives < self.last_lives:
            # Life was lost, override action to press start key (14)
            self.last_lives = current_lives  # Update the lives count
            return 14  # Press start key

        self.last_lives = current_lives  # Update the lives count
        return action
class LoggingWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(LoggingWrapper, self).__init__(env)
        self.last_action = None

    def action(self, action):
        self.last_action = action
        return action

    def step(self, action):
        self.last_action = self.action(action)
        return self.env.step(self.last_action)

class TimeDecayRewardWrapper(gym.Wrapper):
    def __init__(self, env, decay_rate=0.01):
        super().__init__(env)
        self.decay_rate = decay_rate
        self.step_count = 0

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        reward -= self.decay_rate * self.step_count
        return next_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)


class RandomEpsilonWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super(RandomEpsilonWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return action

class BaseActionWrapper(gym.ActionWrapper):
    def __init__(self, env, default_action=0):
        super(BaseActionWrapper, self).__init__(env)
        self.default_action = default_action
        self.first_step = True

    def action(self, action):
        if self.first_step:
            self.first_step = False
            return 14
        if action is None:
            return self.default_action
        return action
    
    def reset(self, **kwargs):
        self.first_step = True
        return self.env.reset(**kwargs)

class NoFrameSkipWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        if hasattr(env.unwrapped, '_frameskip'):
            env.unwrapped._frameskip = 1

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
