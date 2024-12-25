from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq": 350,
        "d_model": 512,
        "datasource": 'opus_book', # The name of the dataset source
        "lang_src": 'en',
        "lang_tgt": 'it',
        "model_folder": "weights", # The folder where the model weights are saved   
        "model_basename": "tmodel_",
        "preload": "latest", # Specifies whether to load the latest weights
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }
    
# The get_weights_file_path function constructs the file path for the model weights based on the configuration and the epoch number.
def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pth"
    return str(Path('.') / model_folder / model_filename)

# The latest_weights_file_path function finds the latest weights file in the specified model folder.
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    
    if len(weights_files) == 0:
        return None
    
    weights_files.sort()
    return str(weights_files[-1])