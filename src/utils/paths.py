import os



full_path = os.path.abspath(__file__)
BASE_DIR_PATH = full_path.split("src")[0] + "src/"


DATA_PATH = BASE_DIR_PATH + "../data/"
MOVIE_LENS_PATH = DATA_PATH + "ml-latest-small/"
EXTRA_DATA_PATH = DATA_PATH + "extra/"
MODELS_CHECKPOINT_DIR = BASE_DIR_PATH + "models/checkpoints/"
EXPERIMENTS_DIR = BASE_DIR_PATH + "experiments/"
EXPERIMENTS_CONFIGS_DIR = EXPERIMENTS_DIR + "configs/"
EXPERIMENTS_PLOTS_DIR = EXPERIMENTS_DIR + "plots/"
EXPERIMENTS_MODELS_DIR = EXPERIMENTS_DIR + "models/"


if not os.path.exists(MODELS_CHECKPOINT_DIR):
    os.makedirs(MODELS_CHECKPOINT_DIR)

if not os.path.exists(EXPERIMENTS_DIR):
    os.makedirs(EXPERIMENTS_DIR)
    if not os.path.exists(EXPERIMENTS_CONFIGS_DIR):
        os.makedirs(EXPERIMENTS_CONFIGS_DIR)
    if not os.path.exists(EXPERIMENTS_PLOTS_DIR):
        os.makedirs(EXPERIMENTS_PLOTS_DIR)
    if not os.path.exists(EXPERIMENTS_MODELS_DIR):
        os.makedirs(EXPERIMENTS_MODELS_DIR)
