import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import numpy as np

class TransformImage(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
    ):
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(env.observation_space.shape[0]*env.observation_space.shape[3], env.observation_space.shape[1], env.observation_space.shape[2]), dtype=np.uint8
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.transpose(obs, (0, 3, 1, 2))
        obs = np.reshape(obs, (obs.shape[0]*obs.shape[1], obs.shape[2], obs.shape[3]))
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        kwargs['seed'] = 45
        obs, info = self.env.reset(**kwargs)
        obs = np.transpose(obs, (0, 3, 1, 2))
        obs = np.reshape(obs, (obs.shape[0]*obs.shape[1], obs.shape[2], obs.shape[3]))
        return obs, info

class TransformAction(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
    ):
        gym.ActionWrapper.__init__(self, env)

        self.action_space = gym.spaces.Box(
            np.array([-1, -1]).astype(np.float32),
            np.array([+1, +1]).astype(np.float32),
        )

    def action(self, action):
        steer, acc = action.astype(np.float64)
        if acc >= 0.0:
            gas = acc
            brake = 0.0
        else:
            gas = 0.0
            brake = np.abs(acc)
        action = np.array([steer, gas, brake])

        return action

class EpisodeReward(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
    ):
        gym.ObservationWrapper.__init__(self, env)
        self.episode_reward = 0

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        self.episode_reward += reward
        if terminated:
            info['episode'] = {'r': self.episode_reward}
        return obs, reward, terminated, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.episode_reward = 0
        return obs


