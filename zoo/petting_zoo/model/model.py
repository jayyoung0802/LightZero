import torch.nn as nn
import torch
from ding.model.common import FCEncoder

class PettingZooEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.agent_encoder = FCEncoder(obs_shape=18, hidden_size_list=[256, 256], activation=nn.ReLU(), norm_type=None)
        self.global_encoder = FCEncoder(obs_shape=48, hidden_size_list=[256, 256], activation=nn.ReLU(), norm_type=None)

    def forward(self, x):
        x_agent = x['agent_state']
        x_agent = self.agent_encoder(x_agent)
        x_global = x['global_state']
        x_global = self.global_encoder(x_global)
        y = torch.concat((x_agent,x_global),dim=1)
        return y