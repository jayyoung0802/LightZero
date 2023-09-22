import torch.nn as nn
import torch
from ding.model.common import FCEncoder

class PettingZooEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.agent_encoder = FCEncoder(obs_shape=18, hidden_size_list=[256, 256], activation=nn.ReLU(), norm_type=None)
        self.global_encoder = FCEncoder(obs_shape=30, hidden_size_list=[256, 256], activation=nn.ReLU(), norm_type=None)

    def forward(self, x):
        agent_state = x['global_state'][:,:18] # [24, 18]
        global_state = x['global_state'][:,18:][::3,] # [8, 30]
        agent_state =  self.agent_encoder(agent_state) # [24, 256]
        global_state = self.global_encoder(global_state) # [8, 256]
        global_state = global_state.unsqueeze(1) # [8, 1, 256]
        global_state = global_state.expand(-1, 3, -1) # [8, 3, 256]
        global_state = global_state.reshape(-1, 256)  # [24, 256]
        return torch.cat((agent_state, global_state), dim=1) # [24, 512]