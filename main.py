# project/main.py

from diffusion_pipe_ui.interface import build_interface
from diffusion_pipe_ui.config import IS_RUNPOD, MAX_UPLOAD_SIZE_MB, BASE_DATASET_DIR, CONFIG_HISTORY_DIR, OUTPUT_DIR, MODEL_DIR

def main():
    demo = build_interface()

    # Ajustar max_file_size baseado no IS_RUNPOD
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "auth": None,
        "share": False,
        "allowed_paths": [BASE_DATASET_DIR, CONFIG_HISTORY_DIR, OUTPUT_DIR, MODEL_DIR, "."]
    }

    if IS_RUNPOD:
        launch_kwargs["max_file_size"] = f"{MAX_UPLOAD_SIZE_MB}mb"  # Gradio espera tamanho em MB
    else:
        launch_kwargs["max_file_size"] = None  # Sem limite

    demo.launch(**launch_kwargs)

if __name__ == "__main__":
    main()
