from .base import SOCRunner
from .policy import SOCForwardPolicyRunner, SOCSeqPolicyRunner
from .supervised import SOCSupervisedSeqRunner, SOCSupervisedForwardRunner
from .text_policy import SOCTextForwardPolicyRunner

__all__ = [
    "SOCRunner",
    "SOCSupervisedSeqRunner",
    "SOCSupervisedForwardRunner",
    "SOCSeqPolicyRunner",
    "SOCForwardPolicyRunner",  # With Text
    'SOCTextForwardPolicyRunner',
]


def make_runner(config):
    if config.runner_name in __all__:
        return globals()[config.runner_name](config)
    else:
        raise Exception('The runner name {} does not exist'.format(config.runner_name))


def get_runner_class(config):
    if config.runner_name in __all__:
        return globals()[config.runner_name]
    else:
        raise Exception('The runner name {} does not exist'.format(config.runner_name))


def get_runners_list():
    return __all__
