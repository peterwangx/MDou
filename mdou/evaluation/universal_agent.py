import torch
import numpy as np

from mdou.env.env import _get_obs_universal, _load_model

class UniversalAgent:
    def __init__(self, position, model_path, evaluate_device_cpu):
        self.evaluate_device_cpu = evaluate_device_cpu
        self.model = _load_model(position, model_path, 'UniversalAgent', evaluate_device_cpu)

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        obs = _get_obs_universal(infoset)

        x_batch = torch.from_numpy(obs['x_batch']).float()

        if torch.cuda.is_available() and not self.evaluate_device_cpu:
            x_batch= x_batch.cuda()
        y_pred = self.model.forward(x_batch, return_value=True)['values']
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]

        return best_action
