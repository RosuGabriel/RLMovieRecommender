#%%
# Imports 
import sys
sys.path.append('../')
from data_processing.dataset import MovieLensDataset
from utils.helpers import show_progress_bar, save_experiment_configuration
from utils.paths import EXPERIMENTS_MODELS_DIR, EXPERIMENTS_PLOTS_DIR
from rl_components.environment import MovieLensEnv
from rl_components.agent import Agent
import time
from datetime import datetime



#%%
# Environment settings
minSteps = 3
maxSteps = 10
keepUserProfiles = True
updateFactor = 0.1

#%%
# Dataset and Environment initialization
dataset = MovieLensDataset(includeSyntheticRatings=True, loadRatings=True, loadMovieEmbeddings=True)
env = MovieLensEnv(dataset=dataset, minSteps=minSteps, maxSteps=maxSteps, keepUserProfiles=keepUserProfiles, 
                   updateFactor=updateFactor)

#%%
# Agent settings
alpha = 0.001
beta  = 0.0001
stateDim  = env.userEmbedding.shape[1] # 403
actionDim = env.allMovieEmbeddings[0].shape[0] # 403
actorHiddenDim  = 768
criticHiddenDim = 512
device = 'cpu'

#%%
# Agent initialization
agent = Agent(alpha=alpha, beta=beta, stateDim=stateDim, actionDim=actionDim,
              actorHiddenDim=actorHiddenDim, criticHiddenDim=criticHiddenDim,
              device=device)

#%%
# Training settings
minutes   = 30
batchSize = 32

lambda_ = None
gamma   = 0.8

EPOCHS_PER_MINUTE = 1300
oneMinuteEpochs = EPOCHS_PER_MINUTE / batchSize
epochs = int(oneMinuteEpochs * minutes)


#%%
# Training functions
def run_simulation():
    currentState, info = env.reset()
    states, actions, rewards, logProbs, values, ratings = [], [], [], [], [], []

    for experience in range(batchSize):
        action, logProb, value, entropy = agent.choose_action(currentState)
        nextState, reward, done, info = env.step(action)

        movieIndex = dataset.get_movie_index(info['movieId'])
        agent.actionHistory.append(movieIndex)
        agent.entropyHistory.append(entropy.item())
        movieFrequency = agent.actionHistory[-10:].count(movieIndex)

        # Penalty on repeated actions
        rating = info['rating'] * (1 - movieFrequency * 0.075)
        reward *= 1 - movieFrequency * 0.075

        currentState = nextState

        if done:
            currentState, info = env.reset()
            done = False

        ratings.append(rating)
        states.append(currentState)
        actions.append(action)
        rewards.append(reward)
        logProbs.append(logProb)
        values.append(value)

    agent.stateValueHistory.append(sum(values) / len(values) if values else 0)
    avgRating = sum(ratings) / len(ratings) if ratings else 0
    agent.avgRatingHistory.append(avgRating)

    return rewards, logProbs, values, avgRating

    
def train(saveFileName):
    bestRating = 0
    for epoch in range(epochs):
        rewards, logProbs, values, avgRating = run_simulation()
        
        if avgRating > bestRating:
            bestRating = avgRating
            agent.save_models(saveFileName)

        returns, advantages = agent.compute_returns_and_advantages(rewards=rewards, values=values, gamma=gamma, lambda_=lambda_)
        agent.learn(values, returns, advantages, logProbs)

        show_progress_bar(epoch + 1, epochs)
    print(f"\nBest Average Rating: {bestRating:.4f}")
    
    return bestRating


def save_experiment(experimentName, bestRating, showPlot=False, saveFig=False, windowSize=25):
    save_experiment_configuration(bestRating, minSteps, maxSteps, keepUserProfiles, updateFactor,
                                  alpha, beta, stateDim, actionDim, actorHiddenDim, criticHiddenDim,
                                  device, minutes, batchSize, lambda_, gamma, epochs, experimentName)
    agent.plot_training(window=0, figName=EXPERIMENTS_PLOTS_DIR + experimentName, showPlot=showPlot, saveFig=saveFig)
    agent.plot_training(window=windowSize, figName=EXPERIMENTS_PLOTS_DIR + experimentName, showPlot=showPlot, saveFig=saveFig)


#%%
# Training loop
start = time.time()
now = datetime.now().strftime("%Y-%m-%d_%H-%M")
bestRating = train(now)
end = time.time()
print(f"Training completed in {end - start:.1f} seconds")

#%%
# Plot experiment and log configs
save_experiment(now, bestRating, showPlot=True, saveFig=True, windowSize=agent.advantageHistory.__len__()//20)
