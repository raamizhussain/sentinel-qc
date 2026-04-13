#!/bin/bash
# SENTINEL GCP Migration Setup Script
# Run this on your GCP f1-micro instance after SSH connection

echo "=== SENTINEL GCP Setup Script ==="

# Update system
echo "Updating system..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and venv
echo "Installing Python 3.11..."
sudo apt install python3.11 python3.11-venv git curl -y

# Create virtual environment
echo "Creating virtual environment..."
python3.11 -m venv sentinel-env
source sentinel-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics mlflow qdrant-client open-clip-torch sentence-transformers numpy scipy
pip install flask gunicorn

# Install Docker for Qdrant
echo "Installing Docker..."
sudo apt install docker.io -y
sudo systemctl start docker
sudo usermod -aG docker $USER

# Run Qdrant
echo "Starting Qdrant..."
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# Install gcsfuse for Cloud Storage
echo "Installing gcsfuse..."
echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install google-cloud-sdk-gcsfuse -y

echo "=== Setup Complete ==="
echo "Next steps:"
echo "1. Upload your data/scripts using: gcloud compute scp --recurse /local/path user@instance:~"
echo "2. Mount Cloud Storage: gcsfuse YOUR_BUCKET /mnt/data"
echo "3. Run your SENTINEL scripts"
echo "4. For web demo: python app.py & gunicorn --bind 0.0.0.0:8000 app:app"</content>
<parameter name="filePath">c:\Users\User\sentinel\scripts\gcp_setup.sh