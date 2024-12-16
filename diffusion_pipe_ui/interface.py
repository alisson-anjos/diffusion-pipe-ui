# project/interface.py

import gradio as gr
import os
from diffusion_pipe_ui.dataset import get_existing_datasets, upload_dataset, finalize_dataset, show_media
from diffusion_pipe_ui.training import train_lora, stop_training
from diffusion_pipe_ui.download import download_output_zip, download_dataset_action
from diffusion_pipe_ui.config import (
    BASE_DATASET_DIR, OUTPUT_DIR, MAX_UPLOAD_SIZE_MB, IS_RUNPOD,
    MODEL_DIR, CONFIG_HISTORY_DIR, MAX_MEDIA
)
import json

def build_interface():
    """Build the Gradio interface."""
    with gr.Blocks() as demo:
        gr.Markdown("# LoRA Training Interface for Hunyuan Video")

        # 1. Step 1: Dataset Management
        gr.Markdown("### Step 1: Dataset Management\nChoose to create a new dataset or select an existing one.")

        with gr.Row():
            dataset_option = gr.Radio(
                choices=["Create New Dataset", "Select Existing Dataset"],
                value="Create New Dataset",
                label="Dataset Option"
            )

        # Container para criar novo dataset
        with gr.Row(visible=True, elem_id="create_new_dataset_container") as create_new_container:
            with gr.Column():
                with gr.Row():
                    dataset_name_input = gr.Textbox(
                        label="Dataset Name",
                        placeholder="Enter your dataset name",
                        interactive=True
                    )
                start_dataset_button = gr.Button("Start New Dataset", interactive=False)  # Inicialmente desativado
                upload_files = gr.File(
                    label="Upload Images (.jpg, .png, .gif, .bmp, .webp), Videos (.mp4), Captions (.txt) or a ZIP archive",
                    file_types=[".jpg", ".png", ".gif", ".bmp", ".webp", ".mp4", ".txt", ".zip"],
                    file_count="multiple",
                    type="filepath",  # Alterado de "file" para "filepath"
                    interactive=True,
                    visible=False  # Inicialmente oculto
                )
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                finalize_button = gr.Button("Finalize Dataset", visible=False)  # Inicialmente oculto
                dataset_path_display = gr.Textbox(label="Current Dataset Path", interactive=False)

        # Container para selecionar dataset existente
        with gr.Row(visible=False, elem_id="select_existing_dataset_container") as select_existing_container:
            with gr.Column():
                existing_datasets = gr.Dropdown(
                    choices=[],  # Inicialmente vazio; será atualizado dinamicamente
                    label="Select Existing Dataset",
                    interactive=True
                )
                # Removido 'selected_dataset_display' para evitar duplicidade

        # 2. Media Gallery (Definido antes dos callbacks que a utilizam)
        gr.Markdown("### Media Gallery")
        gallery = gr.Gallery(
            label="Uploaded Media",
            show_label=False,
            elem_id="gallery",
            columns=3,
            rows=2,
            object_fit="contain",
            height="auto"
        )

        # Função para alternar a visibilidade dos containers com base na opção selecionada
        def toggle_dataset_option(option):
            if option == "Create New Dataset":
                # Mostrar container de criação e ocultar container de seleção
                return (
                    gr.update(visible=True),    # Mostrar create_new_container
                    gr.update(visible=False),   # Ocultar select_existing_container
                    gr.update(choices=[]),      # Limpar Dropdown de datasets existentes
                )
            else:
                # Ocultar container de criação e mostrar container de seleção
                datasets = get_existing_datasets()
                return (
                    gr.update(visible=False),   # Ocultar create_new_container
                    gr.update(visible=True),    # Mostrar select_existing_container
                    gr.update(choices=datasets if datasets else []),  # Atualizar Dropdown
                )

        # Evento para alterar a opção de dataset
        dataset_option.change(
            fn=toggle_dataset_option,
            inputs=dataset_option,
            outputs=[create_new_container, select_existing_container, existing_datasets]
        )

        # Funções para lidar com a criação e upload de datasets
        def handle_start_dataset(dataset_name):
            if not dataset_name.strip():
                return (
                    gr.update(value="Please provide a dataset name."), 
                    gr.update(value=None), 
                    gr.update(visible=True),   # Manter botão visível
                    gr.update(visible=False), 
                    gr.update(visible=False),
                    gr.update(value="")        # Limpar dataset_path_display
                )
            dataset_path, message = upload_dataset([], None, "start", dataset_name=dataset_name)
            if "already exists" in message:
                return (
                    gr.update(value=message), 
                    gr.update(value=None), 
                    gr.update(visible=True),   # Manter botão visível
                    gr.update(visible=False), 
                    gr.update(visible=False),
                    gr.update(value="")        # Limpar dataset_path_display
                )
            return (
                gr.update(value=message), 
                dataset_path, 
                gr.update(visible=False), 
                gr.update(visible=True), 
                gr.update(visible=True),
                gr.update(value=dataset_path)  # Atualizar dataset_path_display
            )

        def handle_upload(files, current_dataset):
            updated_dataset, message = upload_dataset(files, current_dataset, "add")
            return updated_dataset, message

        def handle_finalize(current_dataset):
            """Finalize the dataset creation."""
            if not current_dataset or not os.path.exists(current_dataset):
                return (
                    gr.update(value="No dataset to finalize."), 
                    current_dataset, 
                    gr.update(visible=True), 
                    gr.update(visible=False), 
                    gr.update(visible=False),
                    gr.update(value=""),        # Limpar dataset_path_display
                    []                          # Retornar lista vazia para a galeria
                )
            # Finalização do dataset
            message, dataset = finalize_dataset(current_dataset)
            # Obter os arquivos para atualizar a galeria
            media = show_media(dataset)
            # Retornar status, dataset, visibilidade dos botões e atualizar dataset_path_display e a galeria
            return (
                gr.update(value=message), 
                dataset, 
                gr.update(visible=True), 
                gr.update(visible=False), 
                gr.update(visible=False),
                gr.update(value=dataset),  # Atualizar dataset_path_display
                media                       # Atualizar a galeria
            )

        # Função para habilitar/desabilitar o botão "Start New Dataset" com base na entrada
        def toggle_start_button(name):
            if name.strip():
                return gr.update(interactive=True)
            else:
                return gr.update(interactive=False)

        # Evento para habilitar/desabilitar o botão "Start New Dataset"
        dataset_name_input.change(
            fn=toggle_start_button,
            inputs=dataset_name_input,
            outputs=start_dataset_button
        )

        # Estado para acompanhar o dataset atual
        current_dataset_state = gr.State(None)

        # Botão para iniciar novo dataset
        start_dataset_button.click(
            fn=handle_start_dataset,
            inputs=dataset_name_input,
            outputs=[upload_status, current_dataset_state, start_dataset_button, finalize_button, upload_files, dataset_path_display]
        )

        # Upload de arquivos
        upload_files.upload(
            fn=lambda files, current_dataset: handle_upload(files, current_dataset),
            inputs=[upload_files, current_dataset_state],
            outputs=[current_dataset_state, upload_status],
            queue=True
        )

        # Função para lidar com a seleção de dataset existente e atualizar a galeria
        def handle_select_existing(selected_dataset):
            if selected_dataset:
                dataset_path = os.path.join(BASE_DATASET_DIR, selected_dataset)
                media = show_media(dataset_path)
                return dataset_path, media
            return "", []

        # Evento para atualizar o caminho do dataset e a galeria ao selecionar um existente
        existing_datasets.change(
            fn=handle_select_existing,
            inputs=existing_datasets,
            outputs=[dataset_path_display, gallery]  # Atualizar dataset_path_display e gallery
        )

        # Função para atualizar a galeria com base na seleção ou criação do dataset
        def update_gallery(option, new_dataset, dataset_path):
            if option == "Create New Dataset" and new_dataset:
                return show_media(new_dataset)
            elif option == "Select Existing Dataset" and dataset_path:
                return show_media(dataset_path)
            return []

        # Atualizar galeria quando a opção de dataset muda
        dataset_option.change(
            fn=update_gallery,
            inputs=[dataset_option, current_dataset_state, dataset_path_display],
            outputs=gallery
        )

        # Atualizar galeria quando o caminho do dataset muda
        dataset_path_display.change(
            fn=lambda path: show_media(path),
            inputs=dataset_path_display,
            outputs=gallery
        )

        # Botão para finalizar dataset (deve estar definido após 'gallery')
        finalize_button.click(
            fn=handle_finalize,
            inputs=current_dataset_state,
            outputs=[upload_status, current_dataset_state, start_dataset_button, finalize_button, upload_files, dataset_path_display, gallery]
        )

        # 3. Step 2: Training
        gr.Markdown("### Step 2: Training\nConfigure your training parameters and start or stop the training process.")
        with gr.Column():
            # Output directory
            output_dir_box = gr.Textbox(
                label="Output Directory",
                value=OUTPUT_DIR,
                info="Directory for training outputs",
                interactive=False
            )

            # Training parameters
            gr.Markdown("#### Training Parameters")
            with gr.Row():
                with gr.Column(scale=1):
                    epochs = gr.Number(
                        label="Epochs",
                        value=1000,
                        info="Total number of training epochs"
                    )
                    batch_size = gr.Number(
                        label="Batch Size",
                        value=1,
                        info="Batch size per GPU"
                    )
                    lr = gr.Number(
                        label="Learning Rate",
                        value=2e-5,
                        step=0.0001,
                        info="Optimizer learning rate"
                    )
                    save_every = gr.Number(
                        label="Save Every N Epochs",
                        value=2,
                        info="Frequency to save checkpoints"
                    )
                    eval_every = gr.Number(
                        label="Evaluate Every N Epochs",
                        value=1,
                        info="Frequency to perform evaluations"
                    )
                    rank = gr.Number(
                        label="LoRA Rank",
                        value=32,
                        info="LoRA adapter rank"
                    )
                    dtype = gr.Dropdown(
                        label="LoRA Dtype",
                        choices=['float32', 'float16', 'bfloat16', 'float8'],
                        value="bfloat16"
                    )
                    gradient_accumulation_steps = gr.Number(
                        label="Gradient Accumulation Steps",
                        value=4,
                        info="Micro-batch accumulation steps"
                    )

            # Dataset configuration fields
            gr.Markdown("#### Dataset Configuration")
            with gr.Row():
                enable_ar_bucket = gr.Checkbox(
                    label="Enable AR Bucket",
                    value=True,
                    info="Enable aspect ratio bucketing"
                )
                min_ar = gr.Number(
                    label="Minimum Aspect Ratio",
                    value=0.5,
                    step=0.1,
                    info="Minimum aspect ratio for AR buckets"
                )
            with gr.Row():
                max_ar = gr.Number(
                    label="Maximum Aspect Ratio",
                    value=2.0,
                    step=0.1,
                    info="Maximum aspect ratio for AR buckets"
                )
                num_ar_buckets = gr.Number(
                    label="Number of AR Buckets",
                    value=7,
                    step=1,
                    info="Number of aspect ratio buckets"
                )
            frame_buckets = gr.Textbox(
                label="Frame Buckets",
                value="[1, 33, 65]",
                info="Frame buckets as a JSON list. Example: [1, 33, 65]"
            )

            # Dataset duplication e resoluções
            with gr.Row():
                with gr.Column(scale=1):
                    num_repeats = gr.Number(
                        label="Dataset Num Repeats",
                        value=10,
                        info="Number of times to duplicate the dataset"
                    )
                with gr.Column(scale=1):
                    resolutions_input = gr.Textbox(
                        label="Resolutions",
                        value="[512]",
                        info="Resolutions to train on, given as a list. Example: [512] or [512, 768, 1024]"
                    )

            # Optimizer parameters
            gr.Markdown("#### Optimizer Parameters")
            with gr.Row():
                with gr.Column(scale=1):
                    optimizer_type = gr.Textbox(
                        label="Optimizer Type",
                        value="adamw_optimizer",
                        info="Type of optimizer"
                    )
                    betas = gr.Textbox(
                        label="Betas",
                        value="[0.9, 0.99]",
                        info="Betas for the optimizer"
                    )
                    weight_decay = gr.Number(
                        label="Weight Decay",
                        value=0.01,
                        step=0.0001,
                        info="Weight decay for regularization"
                    )
                    eps = gr.Number(
                        label="Epsilon",
                        value=1e-8,
                        step=0.0000001,
                        info="Epsilon for the optimizer"
                    )

            # Additional training parameters
            gr.Markdown("#### Additional Training Parameters")
            with gr.Row():
                with gr.Column(scale=1):
                    gradient_clipping = gr.Number(
                        label="Gradient Clipping",
                        value=1.0,
                        step=0.1,
                        info="Value for gradient clipping"
                    )
                    warmup_steps = gr.Number(
                        label="Warmup Steps",
                        value=100,
                        step=10,
                        info="Number of warmup steps"
                    )
                    eval_before_first_step = gr.Checkbox(
                        label="Evaluate Before First Step",
                        value=True,
                        info="Perform evaluation before the first training step"
                    )
                    eval_micro_batch_size_per_gpu = gr.Number(
                        label="Eval Micro Batch Size Per GPU",
                        value=1,
                        info="Batch size for evaluation per GPU"
                    )
                    eval_gradient_accumulation_steps = gr.Number(
                        label="Eval Gradient Accumulation Steps",
                        value=1,
                        info="Gradient accumulation steps for evaluation"
                    )
                    checkpoint_every_n_minutes = gr.Number(
                        label="Checkpoint Every N Minutes",
                        value=120,
                        info="Frequency to create checkpoints (in minutes)"
                    )
                    activation_checkpointing = gr.Checkbox(
                        label="Activation Checkpointing",
                        value=True,
                        info="Enable activation checkpointing to save memory"
                    )
                    partition_method = gr.Textbox(
                        label="Partition Method",
                        value="parameters",
                        info="Method for partitioning (e.g., parameters)"
                    )
                    save_dtype = gr.Dropdown(
                        label="Save Dtype",
                        choices=['bfloat16', 'float16', 'float32'],
                        value="bfloat16",
                        info="Data type to save model checkpoints"
                    )
                    caching_batch_size = gr.Number(
                        label="Caching Batch Size",
                        value=1,
                        info="Batch size for caching"
                    )
                    steps_per_print = gr.Number(
                        label="Steps Per Print",
                        value=1,
                        info="Frequency to print training steps"
                    )
                    video_clip_mode = gr.Textbox(
                        label="Video Clip Mode",
                        value="single_middle",
                        info="Mode for video clipping (e.g., single_middle)"
                    )

            # Model paths
            gr.Markdown("#### Model Paths")
            with gr.Row():
                with gr.Column(scale=1):
                    transformer_path = gr.Textbox(
                        label="Transformer Path",
                        value=os.path.join(MODEL_DIR, "hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors"),
                        info="Path to the transformer model weights for Hunyuan Video."
                    )
                    vae_path = gr.Textbox(
                        label="VAE Path",
                        value=os.path.join(MODEL_DIR, "hunyuan_video_vae_fp32.safetensors"),
                        info="Path to the VAE model file."
                    )
                    llm_path = gr.Textbox(
                        label="LLM Path",
                        value=os.path.join(MODEL_DIR, "llava-llama-3-8b-text-encoder-tokenizer"),
                        info="Path to the LLM's text tokenizer and encoder."
                    )
                    clip_path = gr.Textbox(
                        label="CLIP Path",
                        value=os.path.join(MODEL_DIR, "clip-vit-large-patch14"),
                        info="Path to the CLIP model directory."
                    )

            # Botões para iniciar treinamento e exibir logs
            with gr.Row():
                with gr.Column(scale=1):
                    train_button = gr.Button("Start Training")
            with gr.Row():
                with gr.Column(scale=1):
                    output = gr.Textbox(
                        label="Output Logs",
                        lines=20,
                        interactive=False,
                        elem_id="log_box"
                    )

        # Inicialização dos Estados
        is_training_state = gr.State(False)
        logs_state = gr.State("")

        # 4. Ação do Botão de Treinamento
        def toggle_training(is_training, logs, dataset_option_selected, new_dataset, dataset_path, output_dir_val, epochs_val, batch_size_val, lr_val, save_every_val, eval_every_val,
                            rank_val, dtype_val, transformer_path_val, vae_path_val, llm_path_val, clip_path_val,
                            optimizer_type_val, betas_val, weight_decay_val, eps_val, gradient_accumulation_steps_val,
                            num_repeats_val, resolutions_input_val, enable_ar_bucket_val, min_ar_val, max_ar_val, num_ar_buckets_val, frame_buckets_val,
                            gradient_clipping_val, warmup_steps_val, eval_before_first_step_val, eval_micro_batch_size_per_gpu_val,
                            eval_gradient_accumulation_steps_val, checkpoint_every_n_minutes_val, activation_checkpointing_val,
                            partition_method_val, save_dtype_val, caching_batch_size_val, steps_per_print_val, video_clip_mode_val):
            if is_training:
                # Parar treinamento
                message = stop_training()
                logs += message + "\n"
                is_training = False
                return (logs, is_training, gr.update(value="Start Training"))
            else:
                # Determinar o caminho do dataset baseado na seleção
                if dataset_option_selected == "Create New Dataset":
                    dataset_path = new_dataset
                else:
                    dataset_path = dataset_path  # Já atualizado pelo 'handle_select_existing'

                if not dataset_path or not os.path.exists(dataset_path):
                    message = "Please upload a valid dataset before starting training."
                    logs += message + "\n"
                    return (logs, is_training, gr.update(value="Start Training"))

                # Analisar resoluções como lista de inteiros
                try:
                    # Tentar analisar usando JSON
                    resolutions = json.loads(resolutions_input_val)
                    if not isinstance(resolutions, list) or not all(isinstance(i, int) for i in resolutions):
                        raise ValueError
                except:
                    # Se a análise falhar, retornar erro
                    message = "Resolutions must be a list of integers. Example: [512] or [512, 768, 1024]"
                    logs += message + "\n"
                    return (logs, is_training, gr.update(value="Start Training"))

                # Analisar frame_buckets como lista de inteiros
                try:
                    frame_buckets_parsed = json.loads(frame_buckets_val)
                    if not isinstance(frame_buckets_parsed, list) or not all(isinstance(i, int) for i in frame_buckets_parsed):
                        raise ValueError
                except:
                    # Se a análise falhar, retornar erro
                    message = "Frame Buckets must be a list of integers. Example: [1, 33, 65]"
                    logs += message + "\n"
                    return (logs, is_training, gr.update(value="Start Training"))

                # Iniciar treinamento
                is_training = True
                logs = ""
                generator = train_lora(
                    dataset_path=dataset_path, 
                    output_dir=output_dir_val, 
                    epochs=epochs_val, 
                    batch_size=batch_size_val, 
                    lr=lr_val, 
                    save_every=save_every_val, 
                    eval_every=eval_every_val,
                    rank=rank_val, 
                    dtype=dtype_val, 
                    transformer_path=transformer_path_val, 
                    vae_path=vae_path_val, 
                    llm_path=llm_path_val, 
                    clip_path=clip_path_val, 
                    optimizer_type=optimizer_type_val,
                    betas=betas_val, 
                    weight_decay=weight_decay_val, 
                    eps=eps_val, 
                    gradient_accumulation_steps=gradient_accumulation_steps_val, 
                    num_repeats=num_repeats_val, 
                    resolutions=resolutions,
                    enable_ar_bucket=enable_ar_bucket_val, 
                    min_ar=min_ar_val, 
                    max_ar=max_ar_val, 
                    num_ar_buckets=num_ar_buckets_val, 
                    frame_buckets=frame_buckets_parsed,
                    gradient_clipping=gradient_clipping_val,
                    warmup_steps=warmup_steps_val,
                    eval_before_first_step=eval_before_first_step_val,
                    eval_micro_batch_size_per_gpu=eval_micro_batch_size_per_gpu_val,
                    eval_gradient_accumulation_steps=eval_gradient_accumulation_steps_val,
                    checkpoint_every_n_minutes=checkpoint_every_n_minutes_val,
                    activation_checkpointing=activation_checkpointing_val,
                    partition_method=partition_method_val,
                    save_dtype=save_dtype_val,
                    caching_batch_size=caching_batch_size_val,
                    steps_per_print=steps_per_print_val,
                    video_clip_mode=video_clip_mode_val
                )
                for log in generator:
                    print(log)
                    logs += log + "\n"
                    yield (logs, is_training, gr.update(value="Stop Training"))
    
        train_button.click(
            fn=toggle_training,
            inputs=[
                is_training_state, logs_state,
                dataset_option, current_dataset_state, dataset_path_display,
                output_dir_box, epochs, batch_size, lr, save_every, eval_every,
                rank, dtype, transformer_path, vae_path, llm_path, clip_path,
                optimizer_type, betas, weight_decay, eps, gradient_accumulation_steps,
                num_repeats, resolutions_input, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets,
                gradient_clipping, warmup_steps, eval_before_first_step, eval_micro_batch_size_per_gpu,
                eval_gradient_accumulation_steps, checkpoint_every_n_minutes, activation_checkpointing,
                partition_method, save_dtype, caching_batch_size, steps_per_print, video_clip_mode
            ],
            outputs=[output, is_training_state, train_button],
            queue=False
        )

        # 5. Step 3: Download Outputs
        gr.Markdown("### Step 3: Download Outputs\nDownload the training results and configurations.")
        with gr.Row():
            download_output_button = gr.Button("Download Output")
            output_file = gr.File(label="Download Output File")
            download_output_button.click(
                fn=download_output_zip,
                inputs=[],
                outputs=output_file
            )
            download_dataset_button = gr.Button("Download Dataset & Configs")
            dataset_file = gr.File(label="Download Dataset & Configs File")
            download_dataset_button.click(
                fn=download_dataset_action,
                inputs=[dataset_path_display, num_repeats, resolutions_input, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets],
                outputs=dataset_file
            )

        # # 6. Auto-Scroll no Log Box
        # gr.HTML("""
        # <script>
        # const logBox = document.getElementById("log_box");
        # const observer = new MutationObserver(function(mutations) {
        #     mutations.forEach(function(mutation) {
        #         logBox.scrollTop = logBox.scrollHeight;
        #     });
        # });
        # observer.observe(logBox, { childList: true, subtree: true, characterData: true });
        # </script>
        # """)

    return demo
