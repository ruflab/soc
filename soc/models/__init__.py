from .resnet18 import resnet18, ResNetConfig
from .resnet18_policy import ResNet18Policy
from .resnet18mean_policy import ResNet18MeanConcatPolicy
from .resnet18mean_policy import ResNet18MeanFFPolicy
from .resnet18mean_policy import ResNet18MeanFFResPolicy
from .resnet18bilstm_policy import ResNet18BiLSTMConcatPolicy
from .resnet18bilstm_policy import ResNet18BiLSTMFFPolicy
from .resnet18bilstm_policy import ResNet18BiLSTMFFResPolicy
from .resnet18_fusion_policy import ResNet18FusionPolicy, ResNetFusionConfig
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
    "ResNet18MeanConcatPolicy",
    "ResNet18MeanFFPolicy",
    "ResNet18MeanFFResPolicy",
    "ResNet18BiLSTMConcatPolicy",
    "ResNet18BiLSTMFFPolicy",
    "ResNet18BiLSTMFFResPolicy",
    "ResNet18FusionPolicy",
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
