import gym
from abc import abstractmethod, ABC
from mentalitystorm import Dispatcher, Observable, OpenCV, Storeable, ActionEncoder
import torch
from pathlib import Path


class Policy(ABC):
    @abstractmethod
    def action(self, observation): raise NotImplementedError


class RandomPolicy():
    def __init__(self, env):
        self.env = env

    def action(self, observation):
        return self.env.action_space.sample()


class Rollout(Dispatcher, Observable):
    def __init__(self, env):
        Dispatcher.__init__(self)
        self.env = env

    def rollout(self, policy, max_timesteps=100):
        observation = self.env.reset()
        for t in range(max_timesteps):
            screen = self.env.render(mode='rgb_array')
            self.updateObserversWithImage('input', screen, 'numpyRGB')
            action = policy.action(observation)
            self.updateObservers('screen_action',(screen, action),{'func':'screen_action'})
            observation, reward, done, info = self.env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        self.endObserverSession()

if __name__ == '__main__':

    device = torch.device("cuda")
    env = gym.make('SpaceInvaders-v4')
    random_policy = RandomPolicy(env)
    rollout = Rollout(env)
    rollout.registerView('input', OpenCV('input'))
    #rollout.registerObserver('input', ImageVideoWriter('data/video/spaceinvaders/','random'))
    #rollout.registerObserver('input', ImageFileWriter('data/images/spaceinvaders/fullscreen', 'input', 16384))
    #cvae = models.ConvVAE.load('conv_run2_cart')

    name = 'onvn'
    #atari_conv = models.AtariConv_v4()
    atari_conv = Storeable.load('GM53H301W5YS38XH')
    atari_conv = atari_conv.eval()
    ae = ActionEncoder(atari_conv, env, 'SpaceInvaders-v4').to(device)
    rollout.registerView('screen_action', ae)


    for i_episode in range(100):
        rollout.rollout(random_policy, max_timesteps=1000)
