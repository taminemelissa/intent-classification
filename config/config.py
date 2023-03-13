# DEVICE
CUDA_DEVICE = 0

# PARAMETERS
SEED = 42
VERSION = 1
FILENAME = f'{ENCODER_PATH}-{DATASET_NAME}-v{VERSION}'

#PATHS
HOME = '/home/onyxia/work/intent-classification'
MODELS = f'{HOME}/models'
DIR_CHECKPOINTS = f'{HOME}/checkpoints'
LOGS = f'{HOME}/logs'

# DATA
DATASET_NAME = 'dyda_da'
CLASS_NUMBER = 4

# MODEL
ENCODER_PATH = 'distilbert-base-uncased'
TOKENIZER_PATH = 'distilbert-base-uncased'


# HYPERPARAMETERS
BATCH_SIZE = 12
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50
