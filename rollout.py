import gym
from mentalitystorm.config import config
from mentalitystorm.observe import ImageViewer
from mentalitystorm.storage import Storeable
from mentalitystorm.atari import ActionEncoder
from mentalitystorm.data_containers import ActionEmbedding
import torch

from mentalitystorm.policies import VCPolicy, Rollout


def main(gym_environment, policy, output_dir):
    global action_encoder, input_viewer
    env = gym.make(gym_environment)
    action_encoder = ActionEncoder(env, gym_environment, ActionEmbedding(env), policy.v).to(config.device())
    rollout = Rollout(env)
    input_viewer = ImageViewer('input', (320, 480), 'numpyRGB')

    def frame(step):
        input_viewer.update(step.screen)

    rollout.register_before_hook(frame)

    def save(args):
        action_encoder.save_session()

    def action_encoder_frame(step):
        episode = step.meta['episode']
        file_path = config.basepath() / gym_environment / output_dir / str(episode)
        step.meta['filename'] = str(file_path)
        action_encoder.update(step)

    rollout.register_step(action_encoder_frame)
    rollout.register_end_session(save)

    for i_episode in range(531, 1000):
        rollout.rollout(policy, max_timesteps=3000, episode=i_episode)


if __name__ == '__main__':

    gym_environment = 'SpaceInvaders-v4'

    visuals = Storeable.load('.\modelzoo\GM53H301W5YS38XH').to(config.device())
    controller = torch.load(r'.\modelzoo\best_model68')
    policy = VCPolicy(visuals, controller)

    output_dir = 'rl_raw_v2'

    main(gym_environment, policy, output_dir)

