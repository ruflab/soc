from .soc_psql import PSQLConfig
from .soc_psql_forward import PSQLForwardConfig
from .soc_preprocessed_seq import PreprocessedSeqConfig
from .soc_preprocessed_forward import PreprocessedForwardConfig
from .soc_psql_seq import SocPSQLSeqDataset, SocPSQLSeqSAToSDataset, SocPSQLSeqSAToSADataset
from .soc_psql_forward import SocPSQLForwardSAToSADataset
from .soc_psql_forward import SocPSQLForwardSAToSAPolicyDataset
from .soc_file_forward import SocFileForwardSAToSAPolicyDataset
from .soc_preprocessed_seq import SocPreprocessedSeqSAToSDataset
from .soc_preprocessed_seq import SocPreprocessedSeqSAToSADataset
from .soc_preprocessed_seq import SocPreprocessedSeqSAToSAPolicyDataset
from .soc_preprocessed_forward import SocPreprocessedForwardSAToSADataset
from .soc_preprocessed_forward import SocPreprocessedForwardSAToSAPolicyDataset
from .soc_preprocessed_forward import SocLazyPreprocessedForwardSAToSADataset
from .soc_preprocessed_forward import SocLazyPreprocessedForwardSAToSAPolicyDataset

__all__ = [
    'PSQLConfig',
    'PSQLForwardConfig',
    'PreprocessedSeqConfig',
    'PreprocessedForwardConfig',
    'SocPSQLSeqDataset',
    'SocPSQLSeqSAToSDataset',
    'SocPSQLSeqSAToSADataset',
    'SocPSQLForwardSAToSADataset',
    'SocPSQLForwardSAToSAPolicyDataset',
    'SocFileForwardSAToSAPolicyDataset',
    'SocPreprocessedSeqSAToSDataset',
    'SocPreprocessedSeqSAToSADataset',
    'SocPreprocessedSeqSAToSAPolicyDataset',
    'SocPreprocessedForwardSAToSADataset',
    'SocPreprocessedForwardSAToSAPolicyDataset',
    'SocLazyPreprocessedForwardSAToSADataset',
    'SocLazyPreprocessedForwardSAToSAPolicyDataset'
]


def make_dataset(config):
    if config.name in __all__:
        dataset_class = globals()[config.name]
        dataset = dataset_class(config)
        return dataset
    else:
        raise Exception('The dataset name {} does not exist'.format(config.name))


def get_dataset_class(config):
    if config.name in __all__:
        dataset_class = globals()[config.name]
        return dataset_class
    else:
        raise Exception('The dataset name {} does not exist'.format(config.name))


def get_datasets_list():
    return __all__
