# Config file - Typical stuff, just an easy way to change some hyperparameters when testing
import torch
import os

#DATA_PATH = "/home/ncostagliola/Jupyter_Notebook/Payloadbyte-Dataset/Payload_data_CICIDS2017.csv" # Dataset Path
DATA_PATH = "/home/ncostagliola/Jupyter_Notebook/ACI-Dataset/ACI-IoT-2023-Payload-training.csv"
OUTPUT_PATH = "output" # Output Path

END_LAYER = False #True for ACI, False for CICIDS2017.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NAME = ''

BENCHMARK_SEL = 1 #domain = 1, class = 2, gradual = 3

MAX_CLASS = 5000
MIN_CLASS = 1000

TRAIN_MB_SIZE = 100
TRAIN_EPOCHS = 10
EVAL_MB_SIZE = 100
LR = 0.005

USE_TTA = False  # Whether to use Test-Time Adaptation during evaluation
WEIGHT_SHIFT_FACTOR = 0
LOGIT_SHIFT_FACTOR = 0  # Factor to shift logits during test-time adaptation
NORMALIZE_WEIGHTS = False  # Whether to normalize weights after shifting

STRATEGY = "Replay" #"Replay" or "Naive"
PLUGIN = False # Whether to use the custom plugin or not
MEM_SIZE = 100 # Memory size for the Replay strategy

def save_config_details(run_dir):
    """Save configuration details to a text file in the run directory."""
    config_content = f"""Configuration Details:
    =====================
    Data Path: {DATA_PATH}
    Output Path: {OUTPUT_PATH}
    End Layer: {END_LAYER}
    Device: {DEVICE}
    Benchmark Selection: {BENCHMARK_SEL}
    Max Class Size: {MAX_CLASS}
    Min Class Size: {MIN_CLASS}
    Training Mini-batch Size: {TRAIN_MB_SIZE}
    Training Epochs: {TRAIN_EPOCHS}
    Evaluation Mini-batch Size: {EVAL_MB_SIZE}
    Learning Rate: {LR}
    Use Test-Time Adaptation: {USE_TTA}
    Weight Shift Factor: {WEIGHT_SHIFT_FACTOR}
    Logit Shift Factor: {LOGIT_SHIFT_FACTOR}
    Normalize Weights: {NORMALIZE_WEIGHTS}
    Strategy: {STRATEGY}
    Plugin: {PLUGIN}
    Memory Size: {MEM_SIZE}
    """
    
    with open(os.path.join(run_dir, 'config_details.txt'), 'w') as f:
        f.write(config_content)