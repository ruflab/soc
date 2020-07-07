from .soc_psql_seq import SocPSQLSeqDataset, SocPSQLSeqSAToSDataset
from .soc_psql_forward import SocPSQLForwardSAToSADataset
from .soc_preprocessed_seq import SocPreprocessedSeqSAToSDataset
from .soc_preprocessed_forward import SocPreprocessedForwardSAToSADataset

__all__ = [
    'SocPSQLSeqDataset',
    'SocPSQLSeqSAToSDataset',
    'SocPSQLForwardSAToSADataset',
    'SocPreprocessedSeqSAToSDataset',
    'SocPreprocessedForwardSAToSADataset'
]


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
