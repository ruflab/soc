from .resnet18 import resnet18, ResNetConfig
from .resnet18_policy import ResNet18Policy
from .resnet18_fusion_policy import ResNet18FusionPolicy
from .conv_lstm import ConvLSTM, ConvLSTMConfig
from .conv_lstm_policy import ConvLSTMPolicy
from .conv3d import Conv3dModel, Conv3dModelConfig
from .conv3d_policy import Conv3dModelPolicy
from .hexa_conv import HexaConv2d

__all__ = [
    "resnet18",
    "ResNet18Policy",
    "ResNet18FusionPolicy",
    "ResNetConfig",
    "ConvLSTM",
    "ConvLSTMConfig",
    "ConvLSTMPolicy",
    "Conv3dModel",
    "Conv3dModelConfig",
    "Conv3dModelPolicy",
    "HexaConv2d",
]


def make_model(config):
    if config.name in __all__:
        return globals()[config.name](config)
    else:
        raise Exception('The model name {} does not exist'.format(config.name))


def get_model_class(config):
    if config.name in __all__:
        return globals()[config.name]
    else:
        raise Exception('The model name {} does not exist'.format(config.name))


def get_models_list():
    return __all__
