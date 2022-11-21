import torch
import numpy as np

from mdou.env.env import get_obs, _load_model

class DeepAgent:

    def __init__(self, position, model_path, evaluate_device_cpu):
        self.evaluate_device_cpu = evaluate_device_cpu
        self.model = _load_model(position, model_path, 'DeepAgent', evaluate_device_cpu)

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        obs = get_obs(infoset) 

        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available() and not self.evaluate_device_cpu:
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]

        return best_action
