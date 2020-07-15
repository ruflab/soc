from .resnet18 import resnet18
from .resnet18_policy import ResNet18Policy
from .conv_lstm import ConvLSTM
from .conv3d import Conv3dModel
from .hexa_conv import HexaConv2d

__all__ = [
    "resnet18",
    "ResNet18Policy",
    "ConvLSTM",
    "Conv3dModel",
    "HexaConv2d",
]


def make_model(config):
    if config['model'] in __all__:
        return globals()[config['model']](config)
    else:
        raise Exception('The model name {} does not exist'.format(config['model']))


def get_model_class(config):
    if config['model'] in __all__:
        return globals()[config['model']]
    else:
        raise Exception('The model name {} does not exist'.format(config['model']))


def get_models_list():
    return __all__
