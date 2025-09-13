#%%
# Imports
from data_processing.dataset import MovieLensDataset
from rl_components.agent import Agent
import torch
import numpy as np
from utils.helpers import load_experiment_configuration



#%%
# Load dataset and calculate all movie embeddings
dataset = MovieLensDataset(includeSyntheticRatings=False, loadRatings=True, loadMovieEmbeddings=True, includeMyRatings=True)
dataset.calculate_all_users_embeddings()

allMovieEmbeddings = np.vstack(dataset.movieEmbeddingsDF['embedding'].values)


#%%
# Load agent and its configuration
modelName = "2025-09-08_16-32_gen1"

agentSettings = load_experiment_configuration(modelName)[2]
del agentSettings["useResizedStateForCritic"]
agent = Agent()
agent.load_config(modelName, agentSettings)


#%%
# Set user
userId = 300

userEmbedding = dataset.get_user_embedding(userId)
userEmbedding = torch.tensor(userEmbedding, dtype=torch.float32)


#%%
# Get recommendations
action, _, _, _, _, _ = agent.actor(userEmbedding)
recommendations = dataset.get_movieId_from_embedding_similarity(action, allMovieEmbeddings, k=10)
dbUser = userId < 601
if dbUser:
    ratings = dataset.get_user_ratings(userId)
for r in recommendations:
    if dbUser and r in ratings:
        print(dataset.get_movie_title(r),"-",ratings[r])
    elif not dbUser:
        print(dataset.get_movie_title(r))
