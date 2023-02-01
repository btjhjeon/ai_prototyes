import os
from omegaconf import OmegaConf


def load_config(model_path):
    base_config = OmegaConf.load('ai_prototypes/inpainting/lama/configs/prediction/default.yaml')
    model_config = OmegaConf.load(os.path.join(model_path, 'config.yaml'))
    config = OmegaConf.merge(base_config, model_config)

    config.model.path = model_path
    config.training_model.predict_only = True
    config.visualizer.kind = 'noop'
    return config
