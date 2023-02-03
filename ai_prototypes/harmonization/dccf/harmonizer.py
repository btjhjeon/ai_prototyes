import random
from albumentations import Resize

from .data.transforms import HCompose
from .inference.predictor_upsample_hsl import PredictorUpsampleHSL
from .inference.predictor_upsample_hsl_nobackbone import PredictorUpsampleHSLNoBackbone
from .inference import utils as inf_utils


def load_model(model_type, checkpoint_path, device, version='hsl', use_flip=False):
    checkpoint_path = inf_utils.find_checkpoint("", checkpoint_path)
    model = Harmonizer(model_type, checkpoint_path, device, version, use_flip)
    return model


class Harmonizer():
    def __init__(self, model_type, checkpoint_path, device, version='hsl', use_flip=False):
        self.model = inf_utils.load_model(model_type, checkpoint_path, verbose=False)
        if version == 'hsl':
            self.predictor = PredictorUpsampleHSL(self.model, device, with_flip=use_flip)
        elif version == 'hsl_nobb':
            self.predictor = PredictorUpsampleHSLNoBackbone(self.model, device, with_flip=use_flip)
        self.transform = HCompose([Resize(256, 256)])

    def __call__(self, image, mask):
        image_lowres, mask_lowres = self.augment_sample(image, mask)

        pred, _, _ = self.predictor.predict(
            image_lowres,
            None,
            image,
            None,
            None,
            mask_lowres,
            None,
            return_numpy=False
        )
        return pred.cpu().numpy()

    def augment_sample(self, image, mask):
        if self.transform is None:
            return image, mask

        valid_augmentation = False
        while not valid_augmentation:
            aug_output = self.transform(image=image, object_mask=mask)
            valid_augmentation = check_augmented_sample(aug_output)

        return aug_output['image'], aug_output['object_mask']


def check_augmented_sample(aug_output, keep_background_prob=0.0):
    if keep_background_prob < 0.0 or random.random() < keep_background_prob:
        return True

    return aug_output['object_mask'].sum() > 1.0


