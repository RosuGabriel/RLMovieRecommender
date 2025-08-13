# Imports
import sys
sys.path.append("../")
import torch.nn as nn
from torch.distributions import Normal



# Network Classes
class ActorNet(nn.Module):
    def __init__(self, stateDim, actionDim, hiddenDim):
        super(ActorNet, self).__init__()
        self.relu = nn.ReLU()

        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        self.fcLayer1 = nn.Linear(stateDim, hiddenDim)
        self.fcLayer2 = nn.Linear(hiddenDim, hiddenDim)

        self.meanLayer = nn.Linear(hiddenDim, actionDim)
        self.stdLayer = nn.Linear(hiddenDim, actionDim)


    def forward(self, state):
        x = self.relu(self.fcLayer1(state))
        #x = self.relu(self.fcLayer2(x))

        mu = self.meanLayer(x)
        sigma = self.softplus(self.stdLayer(x)) + 1e-6

        dist = Normal(mu, sigma)
        action = dist.sample()

        logProb = dist.log_prob(action).sum(dim=-1)

        return action, logProb



class CriticNet(nn.Module):
    def __init__(self, stateDim, hiddenDim):
        super(CriticNet, self).__init__()
        self.relu = nn.ReLU()

        self.fcLayer1 = nn.Linear(stateDim, hiddenDim)
        self.fcLayer2 = nn.Linear(hiddenDim, 1)


    def forward(self, state):
        x = self.relu(self.fcLayer1(state))
        value = self.fcLayer2(x)

        return value
