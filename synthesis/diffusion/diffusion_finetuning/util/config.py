import torch
import os

# --- SETUP ---
# general
ITERATION = '106'
OUTPUT_TO_FILE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATH = 'augmented_data'
RESULTS_PATH = f"results{ITERATION}"
OUTPUTS_PATH = f"results{ITERATION}/outputs"
MODELS_PATH = f"results{ITERATION}/models"

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(OUTPUTS_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

FIG_PATH = f"{OUTPUTS_PATH}/fig{ITERATION}_"
MODEL_PATH = f"{MODELS_PATH}/model{ITERATION}"
LOGS_PATH = f"{OUTPUTS_PATH}/logs{ITERATION}.txt"

# data info
NUM_CLASSES = 7
IMG_WIDTH = 256
IMG_HEIGHT_SCALED = 94
IMG_HEIGHT = 128
PADDING = 17

# hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
LEARNING_RATE_DISCR = 1e-4
TIMESTEPS = 1000
TIME_EMBEDDING_DIM = 128
BASE_CHANNELS = 128
CLASS_EMB_WEIGHT = 2
NUM_LAYERS = 5

CLASS_FREE_EPOCHS = 100
CLASS_EMB_EPOCHS = 300
DISCR_EPOCHS = 10
FINETUNE_EPOCHS = 100