# Imports
import sys
sys.path.append("../")
import pandas as pd
import re
import numpy as np
from utils.paths import EXPERIMENTS_MODELS_DIR, EXPERIMENTS_CONFIGS_DIR, MODELS_CHECKPOINT_DIR
import json
import shutil


# DataProcessor Class
class DataProcessor:
    # Title and year extraction function  
    def split_title_year(title):
        match = re.match(r"^(.*)\s\((\d{4})\)\s*$", title)
        if match:
            name, year = match.groups()
            return pd.Series([name, int(year)])
        else:
            raise ValueError(f"Title '{title}' does not match the expected format 'Title (Year)'")


    # Genres one-hot encoding function
    def genres_to_vector(movieGenres, allGenres):
        return [1.0 if genre in movieGenres else 0.0 for genre in allGenres]


    # Title embeddings function
    def titles_to_embeddings(titles: list = None, printShape: bool = True, showProgressBar: bool = True):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')

        embeddings = model.encode(titles, show_progress_bar=showProgressBar)
        if printShape:
            print("Title embeddings shape: ", embeddings.shape)

        return embeddings


    # Year normalization function
    def normalize_years(years: list = None):
        if years is None:
            return None
        
        minYear = np.min(years)
        maxYear = np.max(years)
        
        return (years - minYear) / (maxYear - minYear)


    # Rating / Reward converter functions
    def rating_to_reward(rating):
        return 2 * (rating - 0.5) / 4.5 - 1

    def reward_to_rating(reward):
        return (reward + 1) * 4.5 / 2 + 0.5
    

    # Verify if user profile exceeds bounds
    def array_exceeds_bounds(userProfile, minValue, maxValue):
        return np.any(userProfile < minValue) or np.any(userProfile > maxValue)



# Function to print progress bar on a single line
def show_progress_bar(current, total, barLength=50):
    percent = (current / total) * 100
    filled = int(barLength * current / total)
    bar = '█' * filled + '-' * (barLength - filled)
    print(f"\r|{bar}| {percent:.1f}%", end='', flush=True)


def save_experiment_configuration(bestRating, minSteps, maxSteps, keepUserProfiles, updateFactor,
                                  alpha, beta, stateDim, actionDim, actorHiddenDim, criticHiddenDim,
                                  device, minutes, batchSize, lambda_, gamma, epochs, experimentName):
    try:
        config = {
            "bestScore": bestRating,
            "minSteps": minSteps,
            "maxSteps": maxSteps,
            "keepUserProfiles": keepUserProfiles,
            "updateFactor": updateFactor,
            "alpha": alpha,
            "beta": beta,
            "stateDim": stateDim,
            "actionDim": actionDim,
            "actorHiddenDim": actorHiddenDim,
            "criticHiddenDim": criticHiddenDim,
            "device": device,
            "minutes": minutes,
            "batchSize": batchSize,
            "lambda": lambda_,
            "gamma": gamma,
            "epochs": epochs
        }

        with open(EXPERIMENTS_CONFIGS_DIR + f'{experimentName}.json', 'w') as f:
            json.dump(config, f, indent=4)

        shutil.copy2(MODELS_CHECKPOINT_DIR + f'{experimentName}_actor.pth', EXPERIMENTS_MODELS_DIR + f'{experimentName}_actor.pth')
        shutil.copy2(MODELS_CHECKPOINT_DIR + f'{experimentName}_critic.pth', EXPERIMENTS_MODELS_DIR + f'{experimentName}_critic.pth')

        print(f"Experiment configuration and models saved successfully.")
        
    except Exception as e:
        print(f"Error occurred while saving experiment: {e}")
