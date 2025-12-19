import yaml
import os

def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.
    """
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct absolute path to config file
    full_config_path = os.path.join(base_dir, config_path)
    
    if not os.path.exists(full_config_path):
        raise FileNotFoundError(f"Configuration file not found at: {full_config_path}")
        
    with open(full_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Resolve relative paths
    if "paths" in config:
        for key, path in config["paths"].items():
            # If path is relative, make it absolute relative to base_dir
            if not os.path.isabs(path):
                config["paths"][key] = os.path.normpath(os.path.join(base_dir, path))
                
    return config
