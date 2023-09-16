import torch
import torch.nn as nn
from ding.model.common import FCEncoder

class PettingZooEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.agent_num = 3
        self.agent_encoder = FCEncoder(obs_shape=18, hidden_size_list=[256, 256], activation=nn.ReLU(), norm_type=None)
        self.global_encoder = nn.Identity()

    def forward(self, x):
        agent_state = x['global_state'][:,:18]
        global_state = x['global_state'][:,18:]
        agent_state = self.agent_encoder(agent_state)
        global_state = self.global_encoder(global_state)
        return torch.cat([agent_state, global_state], dim=1)