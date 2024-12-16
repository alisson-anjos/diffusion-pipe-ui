# project/config.py

import os

# -----------------------------
# Configuration and Constants
# -----------------------------

# Working directories
MODEL_DIR = "/workspace/models"
BASE_DATASET_DIR = "/workspace/datasets"
OUTPUT_DIR = "/workspace/output"
CONFIG_HISTORY_DIR = "/workspace/config_history"

# Create directories if they don't exist
os.makedirs(CONFIG_HISTORY_DIR, exist_ok=True)
os.makedirs(BASE_DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Maximum number of media to display in the gallery
MAX_MEDIA = 50

# Determine if running on Runpod by checking the environment variable
IS_RUNPOD = os.getenv("IS_RUNPOD", "false").lower() == "true"

# Maximum upload size in MB (Gradio expects max_file_size in MB)
MAX_UPLOAD_SIZE_MB = 500 if IS_RUNPOD else None  # 500MB or no limit
