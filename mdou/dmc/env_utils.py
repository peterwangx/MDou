"""
Here, we wrap the original environment to make it easier
to use. When a game is finished, instead of mannualy reseting
the environment, we do it automatically.
"""
import numpy as np
import torch 

def _format_observation(obs, device):
    """
    A utility function to process observations and
    move them to CUDA.
    """
    position = obs['position']
    if not device == "cpu":
        device = 'cuda:' + str(device)
    device = torch.device(device)
    x_batch = torch.from_numpy(obs['x_batch']).to(device)
    x_no_action = torch.from_numpy(obs['x_no_action'])

    obs = {'x_batch': x_batch,
           'legal_actions': obs['legal_actions'],
           'is_split':obs['is_split'],
    }

    return position, obs, x_no_action

class Environment:
    def __init__(self, env, device):
        """ Initialzie this environment wrapper
        """
        self.env = env
        self.device = device
        self.episode_return = None

    def initial(self):
        initial_position, initial_obs, x_no_action = _format_observation(self.env.reset(), self.device)
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        initial_done = torch.ones(1, 1, dtype=torch.bool)

        return initial_position, initial_obs, dict(
            done=initial_done,
            episode_return=self.episode_return,
            obs_x_no_action=x_no_action,
            )
        
    def step(self, action):
        obs, reward, done, _ = self.env.step(action)

        if done:
            obs = self.env.reset()
            self.episode_return = torch.zeros(1, 1)

        position, obs, x_no_action = _format_observation(obs, self.device)
        done = torch.tensor(done).view(1, 1)
        
        return position, obs, dict(
            done=done,
            episode_return=reward,
            obs_x_no_action=x_no_action,
            )

    def close(self):
        self.env.close()
