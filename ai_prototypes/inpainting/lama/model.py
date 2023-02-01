import os

from .training.trainers import load_checkpoint


def load_model(predict_config, device):
    checkpoint_path = os.path.join(predict_config.model.path, 
                                   'models', 
                                   predict_config.model.checkpoint)
    model = load_checkpoint(predict_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)
    return model
