# Imports
import sys
sys.path.append("../")
from data_processing.dataset import MovieLensDataset
from utils.helpers import DataProcessor
import gym
import numpy as np
import random



# Env definition
class MovieLensEnv(gym.Env):
    def __init__(self, dataset: MovieLensDataset = None, minSteps: int = 3, maxSteps: int = 10, keepUserProfiles: bool = False,
                 useContinuousActions: bool = True, updateFactor: float = 0.1, rarityBonus: float = 0.1):
        # Initialize data
        super(MovieLensEnv, self).__init__()
        self.dataset = dataset if dataset else MovieLensDataset(loadRatings=True, loadMovieEmbeddings=True)
        self.dataset.calculate_all_users_embeddings()
        self.allMovieEmbeddings = np.vstack(self.dataset.movieEmbeddingsDF['embedding'].values)
        self.keepUserProfiles = keepUserProfiles
        self.useContinuousActions = useContinuousActions
        self.updateFactor = updateFactor
        self.rarityBonus = rarityBonus
        self.minSteps = minSteps
        self.maxSteps = maxSteps
        self.usersProfiles = {}

        # Initialize state
        self.reset()

        # Define action and observation spaces
        if self.useContinuousActions:
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.allMovieEmbeddings[0].shape, dtype=np.float32)
        else:
            self.action_space = gym.spaces.Discrete(len(dataset.moviesDF))

        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.userEmbedding.shape, dtype=np.float32)


    # New state and stepCount reset
    def reset(self, userId=None):
        self.stepCount = 0
        self.lastStep = random.randint(self.minSteps, self.maxSteps)
        self._reset_user(userId)

        # Returns initial state and info
        return np.array(self.userEmbedding), {"userId": self.userId,"userRatings": self.userRatings}
    

    def render(self, mode='human'):
        print(f"User ID: {self.userId}")
        print(f"User embedding: {self.userEmbedding}")


    def _reset_user(self, userId=None):
        # Pick a random user if no userId is provided
        if userId is None:
            self.userId = random.choice(self.dataset.allUsers)
        else:
            self.userId = userId

        # Calculate user embedding
        if self.keepUserProfiles and self.userId in self.usersProfiles:
            self.userEmbedding = self.usersProfiles[self.userId]
        elif self.keepUserProfiles:
            self.usersProfiles[self.userId] = self.dataset.get_user_embedding(self.userId)
            self.userEmbedding = self.usersProfiles[self.userId]
        else:
            self.userEmbedding = self.dataset.get_user_embedding(self.userId)

        # Get user ratings
        self.userRatings = self.dataset.get_user_ratings(self.userId)


    def step(self, action):
        self.stepCount += 1

        # Convert action to movieId
        if self.useContinuousActions:
            movieIds = self.dataset.get_movieId_from_embedding_similarity(action.cpu(), self.allMovieEmbeddings, k=100)
            movieId = None
            for id in movieIds:
                if id in self.userRatings:
                    movieId = id
                    break
        else:
            pass

        # Get movie embedding
        movieEmbedding = self.dataset.get_movie_embedding(movieId)
        if movieEmbedding is None:
            raise ValueError(f"Movie ID {movieId} not found in dataset.")
        
        # Calculate reward
        reward = DataProcessor.rating_to_reward(self.userRatings[movieId])

        # Update user embedding
        if reward:
            self.userEmbedding = (self.userEmbedding * (1-self.updateFactor)) + (movieEmbedding * reward * self.updateFactor)
        self.usersProfiles[self.userId] = self.userEmbedding
        
        done = self.stepCount >= self.lastStep
        info = {"movieEmbedding": movieEmbedding, "movieId": movieId, "rating": self.userRatings[movieId]}

        return np.array(self.userEmbedding), reward, done, info
