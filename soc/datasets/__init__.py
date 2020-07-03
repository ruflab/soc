from .soc_psql_seq import SocPSQLSeqDataset, SocPSQLSeqSAToSDataset
from .soc_psql_forward import SocPSQLForwardDataset

__all__ = [
    'SocPSQLSeqDataset',
    'SocPSQLSeqSAToSDataset',
    'SocPSQLForwardDataset', ]


def make_dataset(config):
    if config['dataset'] in __all__:
        return globals()[config['dataset']](config)
    else:
        raise Exception('The dataset name {} does not exist'.format(config['dataset']))


def get_dataset_class(config):
    if config['dataset'] in __all__:
        return globals()[config['dataset']]
    else:
        raise Exception('The dataset name {} does not exist'.format(config['dataset']))


def get_datasets_list():
    return __all__
