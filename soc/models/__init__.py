from .resnet18 import resnet18, ResNetConfig
from .resnet18_policy import ResNet18Policy
from .resnet18_fusion_policy import ResNet18FusionPolicy, ResNetFusionConfig
from .resnet18_concat_policy import ResNet18MeanConcatPolicy, ResNet18MeanFFPolicy
from .conv_lstm import ConvLSTM, ConvLSTMConfig
from .conv_lstm_policy import ConvLSTMPolicy
from .conv3d import Conv3dModel, Conv3dModelConfig
from .conv3d_policy import Conv3dModelPolicy
from .hexa_conv import HexaConv2d
from .hopfield import Hopfield

__all__ = [
    "ResNetConfig",
    "ConvLSTMConfig",
    "Conv3dModelConfig",
    "ResNetFusionConfig",
    "Hopfield",
    "resnet18",
    "ResNet18Policy",
    "ResNet18FusionPolicy",
    "ResNet18MeanConcatPolicy",
    "ResNet18MeanFFPolicy",
    "ConvLSTM",
    "ConvLSTMPolicy",
    "Conv3dModel",
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
