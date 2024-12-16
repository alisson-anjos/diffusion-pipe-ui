#!/bin/bash

echo "pod started"

if [[ $PUBLIC_KEY ]]
then
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    cd ~/.ssh
    echo $PUBLIC_KEY > authorized_keys
    chmod 700 -R ~/.ssh
    cd /
    service ssh start
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate pyenv

if [[ -z "${HF_TOKEN}" ]] || [[ "${HF_TOKEN}" == "enter_your_huggingface_token_here" ]]
then
    echo "HF_TOKEN is not set"
else
    echo "HF_TOKEN is set, logging in..."
    huggingface-cli login --token ${HF_TOKEN}
fi

# Start nginx as reverse proxy to enable api access
service nginx start

# Start JupyterLab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.allow_origin='*' --NotebookApp.token='' --ServerApp.preferred_dir=/workspace &
echo "JupyterLab started"

# Execute the entrypoint script
bash /entrypoint.sh

sleep infinity
