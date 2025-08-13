import os



full_path = os.path.abspath(__file__)
BASE_DIR_PATH = full_path.split("src")[0] + "src/"


DATA_PATH = BASE_DIR_PATH + "../data/"
MOVIE_LENS_PATH = DATA_PATH + "ml-latest-small/"
EXTRA_DATA_PATH = DATA_PATH + "extra/"
MODELS_CHECKPOINT_DIR = BASE_DIR_PATH + "models/checkpoints/"

if not os.path.exists(MODELS_CHECKPOINT_DIR):
    os.makedirs(MODELS_CHECKPOINT_DIR)
