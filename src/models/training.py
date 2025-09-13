#%%
# Imports 
import sys
sys.path.append('../')
import torch
from data_processing.dataset import MovieLensDataset
from utils.helpers import show_progress_bar, save_experiment_configuration, load_experiment_configuration, DataProcessor
from utils.paths import EXPERIMENTS_PLOTS_DIR
from rl_components.gym_environment import MovieLensEnv
from rl_components.agent import Agent
import time
from datetime import datetime
import numpy as np



#%%
# Experiment loading

#----------------------
experimentTime = "2025-09-12_21-42" # empty if not loading
experimentGeneration = 1
mentions = "comparasion with 09-09_10-53 same config but with gae this time"
#----------------------

experimentName = f"{experimentTime}_gen{experimentGeneration}" if experimentTime else ""
bestRating, envSettings, agentSettings, trainingSettings = load_experiment_configuration(experimentName)


#%%
# Environment settings
if not envSettings:

    #----------------------
    minSteps = 3
    maxSteps = 10
    keepUserProfiles = True
    updateFactor = 0.2
    #----------------------

else:
    minSteps = envSettings["minSteps"]
    maxSteps = envSettings["maxSteps"]
    keepUserProfiles = envSettings["keepUserProfiles"]
    updateFactor = envSettings["updateFactor"]


# Dataset and Environment initialization
dataset = MovieLensDataset(includeSyntheticRatings=True, loadRatings=True, loadMovieEmbeddings=True)
env = MovieLensEnv(dataset=dataset, minSteps=minSteps, maxSteps=maxSteps, keepUserProfiles=keepUserProfiles, updateFactor=updateFactor)


#%%
# Agent settings
if not agentSettings:

    #----------------------
    alpha = 0.0004
    beta = 0.0002

    titleDim = 384
    genresDim = 18
    yearDim = 1

    titleOutDim = None
    genresOutDim = None
    yearOutDim = None
    useResizedStateForCritic = False

    if None not in (titleOutDim, genresOutDim, yearOutDim):
        obsDim = titleOutDim + genresOutDim + yearOutDim
    else:
        obsDim = env.userEmbedding.shape[1]

    if useResizedStateForCritic:
        stateDim = obsDim
    else:
        stateDim = env.userEmbedding.shape[1]

    actionDim = env.allMovieEmbeddings[0].shape[0]
    actorHiddenDim = 256

    stateType = 'user-action' # user / user-action / user-product / user-absDiff / user-action-product / user-action-absDiff
    criticLayersDims = [1664, 512, 1]

    device = 'cpu'
    #----------------------
    
    agent = Agent(alpha=alpha, beta=beta,
                obsDim=obsDim, stateDim=stateDim, actionDim=actionDim,
                actorHiddenDim=actorHiddenDim, criticLayersDims=criticLayersDims,
                titleDim=titleDim, genresDim=genresDim, yearDim=yearDim,
                titleOutDim=titleOutDim, genresOutDim=genresOutDim, yearOutDim=yearOutDim,
                device=device, stateType=stateType)
    generation = 0
    time_ = datetime.now().strftime("%Y-%m-%d_%H-%M")

else:
    useResizedStateForCritic = agentSettings["useResizedStateForCritic"]
    del agentSettings["useResizedStateForCritic"]
    alpha = agentSettings["alpha"]
    beta = agentSettings["beta"]
    obsDim = agentSettings["obsDim"]
    stateDim = agentSettings["stateDim"]
    actionDim = agentSettings["actionDim"]
    actorHiddenDim = agentSettings["actorHiddenDim"]
    criticLayersDims = agentSettings["criticLayersDims"]
    titleDim = agentSettings["titleDim"]
    genresDim = agentSettings["genresDim"]
    yearDim = agentSettings["yearDim"]
    titleOutDim = agentSettings["titleOutDim"]
    genresOutDim = agentSettings["genresOutDim"]
    yearOutDim = agentSettings["yearOutDim"]
    device = agentSettings["device"]
    stateType = agentSettings["stateType"]
    
    agent = Agent()
    agent.load_config(experimentName, agentSettings)
    generation = experimentGeneration
    time_ = experimentTime


#%%
# Training settings
#----------------------
minutes = 20
#----------------------

if not trainingSettings:

    #----------------------
    batchSize = 32
    gamma = 0.99
    lambda_ = 0.95
    entropyCoef = 0.05
    normalizedAdvantages = False
    #----------------------

else:
    batchSize = trainingSettings["batchSize"]
    gamma = trainingSettings["gamma"]
    lambda_ = trainingSettings["lambda"]
    entropyCoef = trainingSettings["entropyCoef"]
    normalizedAdvantages = trainingSettings["normalizedAdvantages"]

EPOCHS_PER_MINUTE = 1300
oneMinuteEpochs = EPOCHS_PER_MINUTE / batchSize
epochs = int(oneMinuteEpochs * minutes)


#%%
# Training functions
def run_simulation():
    currentState, info = env.reset()
    states, actions, rewards, logProbs, values, ratings = [], [], [], [], [], []

    for experience in range(batchSize):
        # Use actor to choose an action
        action, logProb, resizedState = agent.choose_action(currentState)
        # Take action in the environment
        nextState, reward, done, info = env.step(action)

        # Get movie index and get action hisctorical frequency
        movieIndex = dataset.get_movie_index(info['movieId'])
        agent.actionHistory.append(movieIndex)
        movieShortTermFrequency = agent.actionHistory[-10:].count(movieIndex)
        movieLongTermFrequency = agent.actionHistory[-200:-10].count(movieIndex)
        # Penalty on repeated actions
        rating = info['rating'] * (1 - movieShortTermFrequency * 0.075)
        rating = max(0.5, rating * (1 - movieLongTermFrequency * 0.03))
        reward = DataProcessor.rating_to_reward(rating)

        # Add action frequencies to the state
        movieShortTermFrequency = movieShortTermFrequency / 10
        movieLongTermFrequency = movieLongTermFrequency / 90
        
        # Use critic to evaluate the action
        value = agent.evaluate_action(currentState, resizedState, action, useResizedStateForCritic)

        # Move to the next state and change the user if the episode is done
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
    global bestRating
    for epoch in range(epochs):
        rewards, logProbs, values, avgRating = run_simulation()
        
        if avgRating > bestRating:
            bestRating = avgRating
            agent.save_models(saveFileName)

        returns, advantages = agent.compute_returns_and_advantages(rewards=rewards, values=values, gamma=gamma, lambda_=lambda_)
        if normalizedAdvantages:
            advantages = torch.stack(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        agent.learn(values, returns, advantages, logProbs, np.mean(agent.entropyHistory[-batchSize:]), entropyCoef)

        show_progress_bar(epoch + 1, epochs)
    print(f"\nBest Average Rating: {bestRating:.4f}")
    
    return bestRating


# Save experiment function
def save_experiment(experimentName, bestRating, showPlot=False, saveFig=False):
    save_experiment_configuration(bestRating=bestRating, minSteps=minSteps, maxSteps=maxSteps, keepUserProfiles=keepUserProfiles, updateFactor=updateFactor,
                                  alpha=alpha, beta=beta, obsDim=obsDim, stateDim=stateDim, actionDim=actionDim, 
                                  titleDim=titleDim, genresDim=genresDim, yearDim=yearDim, titleOutDim=titleOutDim, genresOutDim=genresOutDim, yearOutDim=yearOutDim,
                                  actorHiddenDim=actorHiddenDim, criticLayersDims=criticLayersDims, device=device, stateType=stateType,
                                  minutes=minutes, batchSize=batchSize, lambda_=lambda_, gamma=gamma, epochs=epochs, entropyCoef=entropyCoef,
                                  useResizedStateForCritic=useResizedStateForCritic, normalizedAdvantages=normalizedAdvantages, experimentName=experimentName, mentions=mentions)
    agent.plot_training(window=False, figName=EXPERIMENTS_PLOTS_DIR + experimentName, showPlot=showPlot, saveFig=saveFig)
    agent.plot_training(window=True, figName=EXPERIMENTS_PLOTS_DIR + experimentName, showPlot=showPlot, saveFig=saveFig)



#%%
# Training loop
start = time.time()
generation += 1
experimentName = time_ + f"_gen{generation}"
bestRating = train(experimentName)
end = time.time()
print(f"Training completed in {(end - start)/60:.1f} minutes")
 
#%%
# Show and save experiment 
save_experiment(experimentName, bestRating, showPlot=True, saveFig=True)

# #%%
# # Zoom in last 100 experiences
# agent.plot_training(window=True, showPlot=True, saveFig=False, sliceStart=-100)
