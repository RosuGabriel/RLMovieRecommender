# Imports
import sys
sys.path.append('../')
import torch
import random
from models.networks import CriticNet, ActorNet
from utils.paths import MODELS_CHECKPOINT_DIR
import matplotlib.pyplot as plt
import numpy as np



# Agent definition
class Agent:
    def __init__(self, alpha: float = 0.0001, beta: float = 0.0001,
                 stateDim: int = 403, actionDim: int = 403, actorHiddenDim: int = 128, criticHiddenDim: int = 128,
                 device: str = 'cuda', experienceBufferSize: int = 64):
        self.alpha = alpha
        self.beta = beta
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.actorHiddenDim = actorHiddenDim
        self.criticHiddenDim = criticHiddenDim
        self.experienceBufferSize = experienceBufferSize
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.buffer = []

        # Initialize histories for plotting
        self.actorLossHistory = []
        self.criticLossHistory = []
        self.advantageHistory = []
        self.stateValueHistory = []
        self.entropyHistory = []
        self.avgRatingHistory = []
        self.actionHistory = []

        # Initialize networks
        self.actor = ActorNet(stateDim, actionDim, actorHiddenDim).to(self.device)
        self.critic = CriticNet(stateDim, criticHiddenDim).to(self.device)
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=self.alpha)
        self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), lr=self.beta)


    # Choose an action based on the current state
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action, logProb, entropy = self.actor(state)
        value = self.critic(state)
        return action, logProb, value, entropy


    def compute_returns_and_advantages(self, rewards, values, nextValue=0, gamma=0.99, lambda_=None):
        # Generalized Advantage Estimation
        if lambda_ is not None:
            values = values + [nextValue]
            advantages = []
            gae = 0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + gamma * values[t+1] - values[t]
                gae = delta + gamma * lambda_ * gae
                advantages.insert(0, gae)
            returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        # Without GAE
        else:
            returns = []
            R = nextValue
            for reward in reversed(rewards):
                R = reward + gamma * R
                returns.insert(0, R)

            advantages = [ret - val for ret, val in zip(returns, values)]

        self.advantageHistory.append(sum(advantages)/len(advantages) if advantages else 0)

        return returns, advantages


    # Update models
    def learn(self, values, returns, advantages, logProbs):
        # Convert to tensors
        values = torch.stack(values).to(self.device)
        returns = torch.tensor(returns).to(self.device)
        advantages = torch.tensor(advantages).to(self.device)
        logProbs = torch.stack(logProbs).to(self.device)

        # Update actor
        self.actorOptimizer.zero_grad()
        actorLoss = -torch.mean(logProbs * advantages)
        actorLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actorOptimizer.step()

        # Update critic
        self.criticOptimizer.zero_grad()
        criticLoss = torch.mean((returns - values) ** 2)
        criticLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.criticOptimizer.step()

        # Store losses
        self.actorLossHistory.append(actorLoss.item())
        self.criticLossHistory.append(criticLoss.item())


    # Save/Load models methods
    def save_models(self, fileName, dirPath=None):
        if dirPath is None:
            dirPath = MODELS_CHECKPOINT_DIR
        torch.save(self.actor.state_dict(), dirPath + fileName + '_actor.pth')
        torch.save(self.critic.state_dict(), dirPath + fileName + '_critic.pth')


    def load_models(self, fileName, dirPath=None):
        if dirPath is None:
            dirPath = MODELS_CHECKPOINT_DIR
        self.actor.load_state_dict(torch.load(dirPath + fileName + '_actor.pth', map_location=self.device, weights_only=True))
        self.critic.load_state_dict(torch.load(dirPath + fileName + '_critic.pth', map_location=self.device, weights_only=True))


    # Plot training method
    def plot_training(self, window:int=0, figName="training_plot", showPlot:bool=True, saveFig:bool=False):
        plotData = [
        ("Actor Loss", self.actorLossHistory, 'mediumpurple', None, '-'),
        ("Critic Loss", self.criticLossHistory, 'steelblue', None, '-'),
        ("Advantages", self.advantageHistory, 'saddlebrown', None, '-'),
        ("Average Rating", self.avgRatingHistory, 'olivedrab', None, '-'),
        ("State Values", self.stateValueHistory, 'darkblue', None, '-'),
        ("Entropy", self.entropyHistory, 'teal', None, '-'),
        ("Actions", self.actionHistory, 'crimson', '.', ''),
        ]

        n = len(plotData)
        cols = 2
        rows = (n + 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
        axes = axes.flatten()

        for i, (title, values, color, marker, linestyle) in enumerate(plotData):
            if len(values) > 0 and torch.is_tensor(values[0]):
                values = [v.detach().cpu().numpy().squeeze() for v in values]

            if window and title != "Actions":
                values = np.convolve(values, np.ones(window)/window, mode='valid')
            else:
                step = max(1, len(values) // 1000)  # max 1000 points
                values = values[::step]

            axes[i].plot(values, label=title, color=color, marker=marker, linestyle=linestyle)
            axes[i].set_title(f"{title}")
            axes[i].grid(True)

        # Delete empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        if showPlot:
            plt.show()
        if saveFig and window:
            plt.savefig(f"{figName}_with_window_{window}.png")
        elif saveFig:
            plt.savefig(f"{figName}.png")
