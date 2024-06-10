import yaml
import os

def load_config(config_path):
  with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
  return config


def create_folders(path):
  if not os.path.exists(path):
    os.makedirs(path)


def hadle_configs(data_config, model_config):
  data_config = load_config(data_config)
  model_config = load_config(model_config)
  if 'folders' in model_config:
    for key in model_config:
      if isinstance(key, str):
        create_folders(model_config[key])
      else:
        parent, child = list(key.items())[0]
        create_folders(model_config[parent][child])

  return data_config, model_config
