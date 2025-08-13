# Imports
import sys
sys.path.append("../")
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.paths import MOVIE_LENS_PATH, EXTRA_DATA_PATH
import torch
import numpy as np
import json
from utils.helpers import DataProcessor, show_progress_bar



# MovieLensDataset Class
class MovieLensDataset:
    def __init__(self, loadRatings: bool = False, includeSyntheticRatings: bool = False, includeMyRatings: bool = False, loadEmbeddings: bool = False):
        self.moviesDF = pd.read_csv(MOVIE_LENS_PATH + 'movies.csv')
        self.ratingEmbeddingsDF = None
        
        if loadRatings:
            self.ratingsDF = pd.read_csv(MOVIE_LENS_PATH + 'ratings.csv')
            if includeSyntheticRatings:
                syntheticRatingsDF = pd.read_csv(EXTRA_DATA_PATH + 'hybridPredictedRatings.csv')
                self.add_ratings(syntheticRatingsDF)
            if includeMyRatings:
                myRatingsDF = pd.read_csv(EXTRA_DATA_PATH + 'myRatings.csv')
                self.add_ratings(myRatingsDF)

        self.moviesDF[['title', 'year']] = self.moviesDF['title'].apply(DataProcessor.split_title_year)

        if loadEmbeddings:
            try:
                self.movieEmbeddingsDF = pd.read_csv(EXTRA_DATA_PATH + 'movieEmbeddings.csv')
                self.movieEmbeddingsDF['embedding'] = self.movieEmbeddingsDF['embedding'].apply(lambda x: np.array(json.loads(x)))
            except FileNotFoundError:
                self.create_movie_embeddings(saveToFile=True)
                            
        else:
            self.movieEmbeddingsDF = None

        self.allUsers = self.ratingsDF['userId'].unique() if loadRatings else None


    def create_movie_embeddings(self, saveToFile: bool = True):
        movieIDs = self.moviesDF['movieId'].values
        titleEmbeddings = DataProcessor.titles_to_embeddings(self.moviesDF['title'].values)
        normalizedYears = DataProcessor.normalize_years(self.moviesDF['year'].values).reshape(-1, 1)
        genresVectors = self.moviesDF['genres'].apply(lambda x: DataProcessor.genres_to_vector(x.split('|'), MovieLensDataset.ALL_GENRES)).values
        genresVectors = np.vstack(genresVectors)
        embeddings = np.concatenate([
                # [:-19]
                titleEmbeddings,
                # [-19]
                normalizedYears,
                # [-18:]
                genresVectors
            ], axis=1)
        
        self.movieEmbeddingsDF = pd.DataFrame({
            'movieId': movieIDs,
            'embedding':  [row for row in embeddings],
        })
        if saveToFile:
            embeddingsToSave = self.movieEmbeddingsDF.copy()
            embeddingsToSave['embedding'] = embeddingsToSave['embedding'].apply(lambda x: json.dumps(x.tolist()))
            embeddingsToSave.to_csv(EXTRA_DATA_PATH + 'movieEmbeddings.csv', index=False)


    def get_movieId_from_index(self, index):
        return self.movieEmbeddingsDF.iloc[index]['movieId']

    
    def get_movieId_from_embedding_similarity(self, embedding, allEmbeddings=None, k=1):
        similarities = torch.tensor(cosine_similarity(embedding, allEmbeddings))
        _, moviesIndex = torch.topk(similarities, k)
        return self.movieEmbeddingsDF.iloc[moviesIndex.squeeze()]['movieId']


    def get_movie_embedding(self, movieId):
        if self.movieEmbeddingsDF is not None:
            embedding = self.movieEmbeddingsDF.loc[self.movieEmbeddingsDF['movieId'] == movieId, 'embedding'].iloc[0]
            return np.array(embedding, dtype=np.float32)


    def add_ratings(self, ratingsDF):
        self.ratingsDF = pd.concat([self.ratingsDF, ratingsDF], ignore_index=True)


    def get_movie_title(self, movieId):
        title = self.moviesDF.loc[self.moviesDF['movieId'] == movieId, 'title'].values[0]
        if title.endswith(", The"):
            return "The " + title[:-5]
        return title
    

    def get_movie_index(self, movieId):
        if self.moviesDF is not None:
            return self.moviesDF.loc[self.moviesDF['movieId'] == movieId].index[0]
        raise ValueError("Movies data not loaded.")


    def get_user_ratings(self, userId):
        if self.ratingsDF is not None:
            userRatings = self.ratingsDF[self.ratingsDF['userId'] == userId]
            return userRatings.set_index('movieId')['rating'].to_dict()
        raise ValueError("Ratings data not loaded. Set loadRatings=True when initializing MovieLensDataset.")


    def get_user_embedding(self, userId):
        if self.usersEmbeddingsDF is not None:
            embedding = self.usersEmbeddingsDF.loc[self.usersEmbeddingsDF['userId'] == userId, 'embedding'].iloc[0]
            return np.array(embedding, dtype=np.float32)


    def calculate_user_embedding(self, userId):
        userRatings = self.get_user_ratings(userId)
        if not userRatings:
            raise ValueError(f"No ratings found for user {userId}")

        if self.ratingEmbeddingsDF is None:
            self.ratingEmbeddingsDF = pd.merge(self.movieEmbeddingsDF, self.ratingsDF, on="movieId")

        userEmbedding = np.zeros((1, self.movieEmbeddingsDF['embedding'].iloc[0].shape[0]), dtype=np.float32)
        for movieId, rating in userRatings.items():
            if movieId in self.ratingEmbeddingsDF['movieId'].values:
                movieEmbedding = self.ratingEmbeddingsDF.loc[self.ratingEmbeddingsDF['movieId'] == movieId, 'embedding'].iloc[0]
                userEmbedding += DataProcessor.rating_to_reward(rating) * movieEmbedding

        return userEmbedding / len(userRatings)


    def calculate_all_users_embeddings(self, recalculate: bool = False):
        if not recalculate:
            try:
                self.usersEmbeddingsDF = pd.read_csv(EXTRA_DATA_PATH + 'usersEmbeddings.csv')
                self.usersEmbeddingsDF['embedding'] = self.usersEmbeddingsDF['embedding'].apply(lambda x: np.array(json.loads(x)))
                print("Users embeddings loaded from file.")
                return
            except FileNotFoundError:
                self.usersEmbeddingsDF = None

        print("Calculating all users embeddings...")
        total = len(self.allUsers)
        userIds = []
        usersEmbeddings = []
        for index, userId in enumerate(self.allUsers):
            userIds.append(userId)
            usersEmbeddings.append(self.calculate_user_embedding(userId))
            show_progress_bar(index + 1, total)

        self.usersEmbeddingsDF = pd.DataFrame({
            'userId': userIds,
            'embedding':  usersEmbeddings,
        })

        usersEmbeddingsToSave = self.usersEmbeddingsDF.copy()
        usersEmbeddingsToSave['embedding'] = usersEmbeddingsToSave['embedding'].apply(lambda x: json.dumps(x.tolist()))
        usersEmbeddingsToSave.to_csv(EXTRA_DATA_PATH + 'usersEmbeddings.csv', index=False)

        print(f"\nUsers embeddings calculated and saved to file.", flush=True)


    ALL_GENRES = ["Action",
            "Adventure",
            "Animation",
            "Children",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western"]
