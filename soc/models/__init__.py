from .resnet18 import resnet18
from .conv_lstm import ConvLSTM

__all__ = [
    "resnet18",
    "ConvLSTM", ]


def make_model(config):
    if config['arch'] in __all__:
        return globals()[config['arch']](config)
    else:
        raise Exception('The model name {} does not exist'.format(config['arch']))


def get_model_class(config):
    if config['arch'] in __all__:
        return globals()[config['arch']]
    else:
        raise Exception('The model name {} does not exist'.format(config['arch']))


def get_models_list():
    return __all__
