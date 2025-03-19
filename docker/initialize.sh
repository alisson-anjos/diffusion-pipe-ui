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

#!/bin/bash
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
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.allow_origin='*' --NotebookApp.token='' --ServerApp.preferred_dir=/workspace --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' &
echo "JupyterLab started"

mkdir -p /workspace

# Check if diffusion-pipe directory exists in /workspace
echo "Copying diffusion-pipe to /workspace..."
cp -rf /diffusion-pipe /workspace/

echo "Copying venv to /workspace/diffusion-pipe/venv..."
cp -rf /.venv /workspace/diffusion-pipe/venv

# echo "Updating diffusion-pipe..."
# cd /workspace/diffusion-pipe 
# git pull --verbose

cp -f /entrypoint.sh /workspace/entrypoint.sh

echo "Starting entrypoint.sh..."

bash /workspace/entrypoint.sh

sleep infinity
