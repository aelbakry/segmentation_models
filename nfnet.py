from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.utils import model_zoo
import timm

pretrained = False

pretrained_settings = {

    'dm_nfnet_f0': {
        'imagenet': {
            'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f0-604f9c3a.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },

    'dm_nfnet_f1': {
        'imagenet': {
            'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f1-fc540f82.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },

    'dm_nfnet_f2': {
        'imagenet': {
            'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f2-89875923.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },

    'dm_nfnet_f3': {
        'imagenet': {
            'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f3-d74ab3aa.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'dm_nfnet_f4': {
        'imagenet': {
            'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f4-0ac5b10b.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'dm_nfnet_f5': {
        'imagenet': {
            'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f5-ecb20ab1.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'dm_nfnet_f6': {
        'imagenet': {
            'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f6-e0f12116.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    }
}


def initialize_pretrained_model(model, num_classes, settings):
    if pretrained:
            # assert num_classes == settings['num_classes'], \
            #     'num_classes should be {}, but is {}'.format(
            #         settings['num_classes'], num_classes)
            model.load_state_dict(model_zoo.load_url(settings['url']), strict=False)
            # model.input_space = settings['input_space']
            # model.input_size = settings['input_size']
            # model.input_range = settings['input_range']
            # model.mean = settings['mean']
            # model.std = settings['std']

def dm_nfnet_f0(num_classes=1000, pretrained='imagenet'):
    model = timm.create_model("dm_nfnet_f0", pretrained=False)
    pretrained='imagenet'
    if pretrained is not None:
        settings = pretrained_settings['dm_nfnet_f0'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model

def dm_nfnet_f1(num_classes=1000, pretrained='imagenet'):
    model = timm.create_model("dm_nfnet_f1", pretrained=False)
    pretrained='imagenet'
    if pretrained is not None:
        settings = pretrained_settings['dm_nfnet_f1'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model

def dm_nfnet_f2(num_classes=1000, pretrained='imagenet'):
    model = timm.create_model("dm_nfnet_f2", pretrained=pretrained)
    pretrained='imagenet'
    if pretrained is not None:
        settings = pretrained_settings['dm_nfnet_f2'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model

def dm_nfnet_f3(num_classes=1000, pretrained='imagenet'):
    model = timm.create_model("dm_nfnet_f3", pretrained=pretrained)
    pretrained='imagenet'
    if pretrained is not None:
        settings = pretrained_settings['dm_nfnet_f3'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model

def dm_nfnet_f4(num_classes=1000, pretrained='imagenet'):
    model = timm.create_model("dm_nfnet_f4", pretrained=pretrained)
    pretrained='imagenet'
    if pretrained is not None:
        settings = pretrained_settings['dm_nfnet_f4'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model

def dm_nfnet_f5(num_classes=1000, pretrained='imagenet'):
    model = timm.create_model("dm_nfnet_f5", pretrained=pretrained)
    pretrained='imagenet'
    if pretrained is not None:
        settings = pretrained_settings['dm_nfnet_f5'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model

def dm_nfnet_f6(num_classes=1000, pretrained='imagenet'):
    model = timm.create_model("dm_nfnet_f6", pretrained=pretrained)
    pretrained='imagenet'
    if pretrained is not None:
        settings = pretrained_settings['dm_nfnet_f6'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
