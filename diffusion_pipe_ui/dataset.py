# project/dataset.py

import os
import shutil
import zipfile
import json
import toml
from diffusion_pipe_ui.config import BASE_DATASET_DIR, CONFIG_HISTORY_DIR, MAX_UPLOAD_SIZE_MB, IS_RUNPOD, MAX_MEDIA
from diffusion_pipe_ui.utils import generate_unique_filename

def get_existing_datasets():
    """Retrieve a list of existing datasets."""
    datasets = [d for d in os.listdir(BASE_DATASET_DIR) if os.path.isdir(os.path.join(BASE_DATASET_DIR, d))]
    return datasets

def create_dataset_config(dataset_path, dataset_name, num_repeats, resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets):
    """Create and save the dataset configuration in TOML format."""
    dataset_config = {
        "dataset_name": dataset_name,
        "resolutions": resolutions,
        "enable_ar_bucket": enable_ar_bucket,
        "min_ar": min_ar,
        "max_ar": max_ar,
        "num_ar_buckets": num_ar_buckets,
        "frame_buckets": frame_buckets,
        "directory": [
            {
                "path": dataset_path,
                "num_repeats": num_repeats
            }
        ]
    }
    dataset_file = generate_unique_filename("dataset_auto")
    dataset_path_full = os.path.join(CONFIG_HISTORY_DIR, dataset_file)
    with open(dataset_path_full, "w") as f:
        toml.dump(dataset_config, f)
    return dataset_path_full

def upload_dataset(files, current_dataset, action, dataset_name=None):
    """
    Handle uploaded dataset files and store them in a unique directory.
    Action can be 'start' (initialize a new dataset) or 'add' (add files to current dataset).
    """
    if action == "start":
        if not dataset_name:
            return current_dataset, "Please provide a dataset name."
        # Ensure the dataset name does not contain invalid characters
        dataset_name = "".join(c for c in dataset_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
        dataset_dir = os.path.join(BASE_DATASET_DIR, dataset_name)
        if os.path.exists(dataset_dir):
            return current_dataset, f"Dataset '{dataset_name}' already exists. Please choose a different name."
        os.makedirs(dataset_dir, exist_ok=True)
        return dataset_dir, f"Started new dataset: {dataset_dir}"

    if not current_dataset:
        return current_dataset, "Please start a new dataset before uploading files."

    if not files:
        return current_dataset, "No files uploaded."

    # Calculate the total size of the current dataset
    total_size = 0
    for root, dirs, files_in_dir in os.walk(current_dataset):
        for f in files_in_dir:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)

    # Calculate the size of the new files
    new_files_size = 0
    for file in files:
        if IS_RUNPOD:
            new_files_size += os.path.getsize(file.name)

    # Check if adding these files would exceed the limit
    if IS_RUNPOD and (total_size + new_files_size) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        return current_dataset, f"Upload would exceed the {MAX_UPLOAD_SIZE_MB}MB limit on Runpod. Please upload smaller files or finalize the dataset."

    uploaded_files = []

    for file in files:
        file_path = file.name
        filename = os.path.basename(file_path)
        dest_path = os.path.join(current_dataset, filename)

        if zipfile.is_zipfile(file_path):
            # If the file is a ZIP, extract its contents
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(current_dataset)
                uploaded_files.append(f"{filename} (extracted)")
            except zipfile.BadZipFile:
                uploaded_files.append(f"{filename} (invalid ZIP)")
                continue
        else:
            # Check if the file is a supported format
            if filename.lower().endswith(('.mp4', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.txt')):
                shutil.copy(file_path, dest_path)
                uploaded_files.append(filename)
            else:
                uploaded_files.append(f"{filename} (unsupported format)")
                continue

    return current_dataset, f"Uploaded files: {', '.join(uploaded_files)}"

def finalize_dataset(current_dataset):
    """Finalize the dataset creation."""
    if not current_dataset or not os.path.exists(current_dataset):
        return "No dataset to finalize.", current_dataset
    # Adicione qualquer passo de finalização necessário aqui
    return f"Dataset {current_dataset} has been finalized.", current_dataset

def show_media(dataset_dir):
    """Display uploaded images and .mp4 videos in a single gallery."""
    if not dataset_dir or not os.path.exists(dataset_dir):
        # Retorna uma lista vazia se o dataset_dir for inválido
        return []

    # Lista de arquivos de imagem e vídeos .mp4
    media_files = [
        f for f in os.listdir(dataset_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.mp4'))
    ]

    # Obtenha os caminhos absolutos dos arquivos
    media_paths = [os.path.abspath(os.path.join(dataset_dir, f)) for f in media_files[:MAX_MEDIA]]

    # Verifique se os arquivos existem
    existing_media = [f for f in media_paths if os.path.exists(f)]

    return existing_media
