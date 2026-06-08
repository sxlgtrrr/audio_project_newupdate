import os
import torch
import numpy as np

SAMPLE_RATE = 16000
DURATION = 3.0
N_MFCC = 40
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 1024

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
NUM_CLASSES = len(EMOTIONS)

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WAV2VEC2_MODEL_NAME = 'facebook/wav2vec2-base'
WAV2VEC2_BATCH_SIZE = 8
WAV2VEC2_LR = 2e-5
WAV2VEC2_EPOCHS = 20

DATA_DIR = './data'
MODEL_SAVE_PATH = './models/best_model.pth'
LOG_DIR = './logs'

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)