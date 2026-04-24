import yaml
import os
import shutil
from datetime import datetime

def load_yaml(file_path):

    with open(file_path, 'r') as file:
        config = yaml.load(file)
    return config

def create_experiment_dir(base_dir="./models", config_dir="./config"):

    # 1. Generate the timestamp (e.g., 2026_04_25_12_04)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    exp_dir = os.path.join(base_dir, timestamp)
    
    # 2. Create the directory 
    os.makedirs(exp_dir, exist_ok=True)
    print(f"📁 Created experiment tracking folder: {exp_dir}")
    
    # 3. Copy the YAML files into the new folder
    config_src = os.path.join(config_dir, "config.yaml")
    model_config_src = os.path.join(config_dir, "model_config.yaml")
    
    shutil.copy(config_src, os.path.join(exp_dir, "config.yaml"))
    shutil.copy(model_config_src, os.path.join(exp_dir, "model_config.yaml"))
    
    return exp_dir