Write-Host "Installing TensorFlow..."
pip3 install nvidia-cudnn-cu11
pip3 install "tensorflow<2.11"

Write-Host "Installing Python packages from pip-requirements.txt..."
pip3 install -r pip-requirements.txt

Write-Host "Installation complete."