# Imports
import sys
sys.path.append("../")
from torch import cat, sigmoid
import torch.nn as nn
from torch.distributions import Normal



# Network Classes
class ActorNet(nn.Module):
    def __init__(self, obsDim, actionDim, hiddenDim,
                 titleDim, genresDim, yearDim, titleOutDim, genresOutDim, yearOutDim):
        super(ActorNet, self).__init__()
        self.titleDim = titleDim
        self.genresDim = genresDim
        self.yearDim = yearDim
        self.titleFC = None
        self.genresFC = None
        self.yearFC = None

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.33)

        if None not in (titleOutDim, genresOutDim, yearOutDim):
            self.titleFC = nn.Linear(titleDim, titleOutDim)
            self.genresFC = nn.Linear(genresDim, genresOutDim)
            self.yearFC = nn.Linear(yearDim, yearOutDim)

        self.sharedLayer1 = nn.Linear(obsDim, hiddenDim*2)

        self.meanLayer1 = nn.Linear(hiddenDim*2, hiddenDim)
        self.stdLayer1 = nn.Linear(hiddenDim*2, hiddenDim)

        self.meanLayer2 = nn.Linear(hiddenDim, actionDim)
        self.stdLayer2 = nn.Linear(hiddenDim, actionDim)


    def forward(self, observation):
        state = observation

        if None not in (self.titleFC, self.genresFC, self.yearFC):
            title = state[:,:self.titleDim]
            title = self.titleFC(title)
            year = state[:,self.titleDim:-self.genresDim]
            year = self.yearFC(year)
            genres = state[:,-self.genresDim:]
            genres = self.genresFC(genres)
            state = cat((title, year, genres), dim=-1)

        x = self.relu(self.sharedLayer1(state))
        x = self.dropout(x)

        mu = self.relu(self.meanLayer1(x))
        mu = self.dropout(mu)
        mu = self.tanh(self.meanLayer2(mu))

        sigma = self.relu(self.stdLayer1(x))
        sigma = self.dropout(sigma)
        sigma = self.stdLayer2(sigma)
        sigma = sigmoid(sigma) * (0.8 - 0.1) + 0.1

        dist = Normal(mu, sigma)
        action = dist.sample()

        logProb = dist.log_prob(action).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return action, logProb, entropy, state, mu, sigma



class CriticNet(nn.Module):
    def __init__(self, layerDims):
        super(CriticNet, self).__init__()
        
        layers = []
        for i in range(len(layerDims)-1):
            layers.append(nn.Linear(layerDims[i], layerDims[i+1]))
            if i < len(layerDims)-2:
                layers.append(nn.ReLU())
                
        self.model = nn.Sequential(*layers)


    def forward(self, state):
        return self.model(state)
