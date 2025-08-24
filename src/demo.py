#%%
from data_processing.dataset import MovieLensDataset
from rl_components.agent import Agent
import torch
import numpy as np


#%%
dataset = MovieLensDataset(includeSyntheticRatings=False, loadRatings=False, loadMovieEmbeddings=True, includeMyRatings=True)
dataset.calculate_all_users_embeddings()
agent = Agent(stateDim=403, actionDim=403, actorHiddenDim=1028, criticHiddenDim=512, device='cpu')
allMovieEmbeddings = np.vstack(dataset.movieEmbeddingsDF['embedding'].values)

#%%
agent.load_models(fileName="2025-08-15_21-58-00")


#%%
userId = 611
userEmbedding = dataset.get_user_embedding(userId)
userEmbedding = torch.tensor(userEmbedding, dtype=torch.float32)

#%%
action, _ = agent.actor(userEmbedding)
recommendations = dataset.get_movieId_from_embedding_similarity(action, allMovieEmbeddings, k=10)
for r in recommendations:
    print(dataset.get_movie_title(r))