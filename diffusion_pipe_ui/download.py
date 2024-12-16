# project/download.py

import os
import zipfile
from diffusion_pipe_ui.config import OUTPUT_DIR, CONFIG_HISTORY_DIR
from diffusion_pipe_ui.dataset import create_dataset_config
import json

def download_output_zip():
    """Create a zip file with the training outputs for download."""
    zip_filename = "output.zip"  # Nome do arquivo ZIP
    zip_path = os.path.join(os.getcwd(), zip_filename)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, OUTPUT_DIR)
                zf.write(filepath, arcname)
    return zip_path

def download_dataset_config_zip(dataset_dir):
    """Create a zip file with the dataset and configurations for download."""
    zip_filename = "dataset_configs.zip"  # Nome do arquivo ZIP
    zip_path = os.path.join(os.getcwd(), zip_filename)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        if dataset_dir and os.path.exists(dataset_dir):
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    arcname = os.path.relpath(filepath, os.path.dirname(dataset_dir))
                    zf.write(filepath, arcname)
        for root, dirs, files in os.walk(CONFIG_HISTORY_DIR):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.join("config_history", os.path.relpath(filepath, CONFIG_HISTORY_DIR))
                zf.write(filepath, arcname)
    return zip_path

def download_dataset_action(dataset_dir, num_repeats, resolutions_input, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets_input):
    """Action para baixar o dataset e configurações."""
    if not dataset_dir or not os.path.exists(dataset_dir):
        return None  # Gradio tratará como nenhum arquivo para baixar
    if not num_repeats:
        num_repeats = 10
    try:
        # Parse resolutions
        resolutions = json.loads(resolutions_input)
        if not isinstance(resolutions, list) or not all(isinstance(i, int) for i in resolutions):
            raise ValueError
    except:
        # Se a análise falhar, use o valor padrão
        resolutions = [512]  # Valor padrão
    try:
        # Parse frame_buckets
        frame_buckets = json.loads(frame_buckets_input)
        if not isinstance(frame_buckets, list) or not all(isinstance(i, int) for i in frame_buckets):
            raise ValueError
    except:
        # Se a análise falhar, use o valor padrão
        frame_buckets = [1, 33, 65]  # Valor padrão
    create_dataset_config(
        dataset_path=dataset_dir, 
        dataset_name=os.path.basename(dataset_dir), 
        num_repeats=num_repeats, 
        resolutions=resolutions, 
        enable_ar_bucket=enable_ar_bucket, 
        min_ar=min_ar, 
        max_ar=max_ar, 
        num_ar_buckets=num_ar_buckets, 
        frame_buckets=frame_buckets
    )
    return download_dataset_config_zip(dataset_dir)
