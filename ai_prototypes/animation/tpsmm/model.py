import yaml
import numpy as np
from scipy.spatial import ConvexHull

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from .inpainting_network import InpaintingNetwork
from .keypoint_detector import KPDetector
from .dense_motion import DenseMotionNetwork
from .avd_network import AVDNetwork
from .util import AntiAliasInterpolation2d, TPS


def load_model(config_path, checkpoint_path, device):
    model = TPSMM(config_path)
    model.to(device)
    model.load_checkpoints(checkpoint_path)
    model.eval()
    return model


class TPSMM():
    def __init__(self, config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        self.kp_detector = KPDetector(**config['model_params']['common_params'])
        self.dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                                       **config['model_params']['dense_motion_params'])
        self.avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                                      **config['model_params']['avd_network_params'])
        self.device = torch.device('cpu')
    
    def to(self, device):
        self.kp_detector.to(device)
        self.dense_motion_network.to(device)
        self.inpainting.to(device)
        self.avd_network.to(device)
        self.device = device

    def load_checkpoints(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
    
        self.inpainting.load_state_dict(checkpoint['inpainting_network'])
        self.kp_detector.load_state_dict(checkpoint['kp_detector'])
        self.dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
        if 'avd_network' in checkpoint:
            self.avd_network.load_state_dict(checkpoint['avd_network'])

    def eval(self):
        self.inpainting.eval()
        self.kp_detector.eval()
        self.dense_motion_network.eval()
        self.avd_network.eval()

    def __call__(self, source, driving, mode='standard'):
        assert mode in ['standard', 'relative', 'avd']

        predictions = []
        with torch.no_grad():
            kp_source = self.kp_detector(source)
            kp_driving_initial = self.kp_detector(driving[:, :, 0])

            for frame_idx in range(driving.shape[2]):
                driving_frame = driving[:, :, frame_idx]
                driving_frame = driving_frame.to(self.device)
                kp_driving = self.kp_detector(driving_frame)
                if mode == 'standard':
                    kp_norm = kp_driving
                elif mode=='relative':
                    kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving,
                                        kp_driving_initial=kp_driving_initial)
                elif mode == 'avd':
                    kp_norm = self.avd_network(kp_source, kp_driving)
                dense_motion = self.dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                        kp_source=kp_source, bg_param = None, 
                                                        dropout_flag = False)
                out = self.inpainting(source, dense_motion)

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        return predictions


def relative_kp(kp_source, kp_driving, kp_driving_initial):

    source_area = ConvexHull(kp_source['fg_kp'][0].data.cpu().numpy()).volume
    driving_area = ConvexHull(kp_driving_initial['fg_kp'][0].data.cpu().numpy()).volume
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_new = {k: v for k, v in kp_driving.items()}

    kp_value_diff = (kp_driving['fg_kp'] - kp_driving_initial['fg_kp'])
    kp_value_diff *= adapt_movement_scale
    kp_new['fg_kp'] = kp_value_diff + kp_source['fg_kp']

    return kp_new


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, bg_predictor, dense_motion_network, inpainting_network, train_params, *kwargs):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.inpainting_network = inpainting_network
        self.dense_motion_network = dense_motion_network

        self.bg_predictor = None
        if bg_predictor:
            self.bg_predictor = bg_predictor
            self.bg_start = train_params['bg_start']

        self.train_params = train_params
        self.scales = train_params['scales']

        self.pyramid = ImagePyramide(self.scales, inpainting_network.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']
        self.dropout_epoch = train_params['dropout_epoch']
        self.dropout_maxp = train_params['dropout_maxp']
        self.dropout_inc_epoch = train_params['dropout_inc_epoch']
        self.dropout_startp =train_params['dropout_startp']
        
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()


    def forward(self, x, epoch):
        kp_source = self.kp_extractor(x['source'])
        kp_driving = self.kp_extractor(x['driving'])
        bg_param = None
        if self.bg_predictor:
            if(epoch>=self.bg_start):
                bg_param = self.bg_predictor(x['source'], x['driving'])
          
        if(epoch>=self.dropout_epoch):
            dropout_flag = False
            dropout_p = 0
        else:
            # dropout_p will linearly increase from dropout_startp to dropout_maxp 
            dropout_flag = True
            dropout_p = min(epoch/self.dropout_inc_epoch * self.dropout_maxp + self.dropout_startp, self.dropout_maxp)
        
        dense_motion = self.dense_motion_network(source_image=x['source'], kp_driving=kp_driving,
                                                    kp_source=kp_source, bg_param = bg_param, 
                                                    dropout_flag = dropout_flag, dropout_p = dropout_p)
        generated = self.inpainting_network(x['source'], dense_motion)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])

        # reconstruction loss
        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
            loss_values['perceptual'] = value_total

        # equivariance loss
        if self.loss_weights['equivariance_value'] != 0:
            transform_random = TPS(mode = 'random', bs = x['driving'].shape[0], **self.train_params['transform_params'])
            transform_grid = transform_random.transform_frame(x['driving'])
            transformed_frame = F.grid_sample(x['driving'], transform_grid, padding_mode="reflection",align_corners=True)
            transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp
        
            warped = transform_random.warp_coordinates(transformed_kp['fg_kp'])
            kp_d = kp_driving['fg_kp']
            value = torch.abs(kp_d - warped).mean()
            loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

        # warp loss
        if self.loss_weights['warp_loss'] != 0:
            occlusion_map = generated['occlusion_map']
            encode_map = self.inpainting_network.get_encode(x['driving'], occlusion_map)
            decode_map = generated['warped_encoder_maps']
            value = 0
            for i in range(len(encode_map)):
                value += torch.abs(encode_map[i]-decode_map[-i-1]).mean()

            loss_values['warp_loss'] = self.loss_weights['warp_loss'] * value
        
        # bg loss
        if self.bg_predictor and epoch >= self.bg_start and self.loss_weights['bg'] != 0:
            bg_param_reverse = self.bg_predictor(x['driving'], x['source'])
            value = torch.matmul(bg_param, bg_param_reverse)
            eye = torch.eye(3).view(1, 1, 3, 3).type(value.type())
            value = torch.abs(eye - value).mean()
            loss_values['bg'] = self.loss_weights['bg'] * value

        return loss_values, generated
