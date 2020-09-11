from .soc_psql import PSQLConfig
from .soc_psql_forward import PSQLForwardConfig
from .soc_preprocessed_seq import PreprocessedSeqConfig
from .soc_preprocessed_forward import PreprocessedForwardConfig
from .soc_preprocessed_text_forward import PreprocessedTextForwardConfig

from .soc_psql_text_seq import PSQLTextConfig
from .soc_psql_text_forward import PSQLTextForwardConfig
from .soc_file_text_seq import FileTextConfig
from .soc_file_text_forward import FileTextForwardConfig

from .soc_psql_seq import SocPSQLSeqDataset
from .soc_psql_seq import SocPSQLSeqSAToSDataset
from .soc_psql_seq import SocPSQLSeqSAToSADataset
from .soc_psql_forward import SocPSQLForwardSAToSADataset
from .soc_psql_forward import SocPSQLForwardSAToSAPolicyDataset
from .soc_preprocessed_seq import SocPreprocessedSeqSAToSDataset
from .soc_preprocessed_seq import SocPreprocessedSeqSAToSADataset
from .soc_preprocessed_seq import SocPreprocessedSeqSAToSAPolicyDataset
from .soc_preprocessed_forward import SocPreprocessedForwardSAToSADataset
from .soc_preprocessed_forward import SocPreprocessedForwardSAToSAPolicyDataset
from .soc_preprocessed_forward import SocLazyPreprocessedForwardSAToSADataset
from .soc_preprocessed_forward import SocLazyPreprocessedForwardSAToSAPolicyDataset

from .soc_psql_text_seq import SocPSQLTextBertSeqDataset
from .soc_psql_text_forward import SocPSQLTextBertForwardSAToSADataset
from .soc_psql_text_forward import SocPSQLTextBertForwardSAToSAPolicyDataset
from .soc_file_seq import SocFileSeqDataset
from .soc_file_text_seq import SocFileTextBertSeqDataset
from .soc_file_text_forward import SocFileTextBertForwardSAToSAPolicyDataset
from .soc_preprocessed_text_forward import SocPreprocessedTextBertForwardSAToSADataset
from .soc_preprocessed_text_forward import SocPreprocessedTextBertForwardSAToSAPolicyDataset

# yapf: disable
__all__ = [
    # Configurations
    'PSQLConfig',
    'PSQLForwardConfig',
    'PreprocessedSeqConfig',
    'PreprocessedForwardConfig',
    'PSQLTextConfig',
    'PSQLTextForwardConfig',
    'PreprocessedTextForwardConfig',
    'FileTextConfig',
    'FileTextForwardConfig',
    # Soc Datasets
    'SocPSQLSeqDataset',
    'SocPSQLSeqSAToSDataset',
    'SocPSQLSeqSAToSADataset',
    'SocPSQLForwardSAToSADataset',
    'SocPSQLForwardSAToSAPolicyDataset',
    'SocPreprocessedSeqSAToSDataset',
    'SocPreprocessedSeqSAToSADataset',
    'SocPreprocessedSeqSAToSAPolicyDataset',
    'SocPreprocessedForwardSAToSADataset',
    'SocPreprocessedForwardSAToSAPolicyDataset',
    'SocLazyPreprocessedForwardSAToSADataset',
    'SocLazyPreprocessedForwardSAToSAPolicyDataset',
    # Soc Dataset with text
    'SocPSQLTextBertSeqDataset',
    'SocPSQLTextBertForwardSAToSADataset',
    'SocPSQLTextBertForwardSAToSAPolicyDataset',
    'SocFileSeqDataset',
    'SocFileTextBertSeqDataset',
    'SocFileTextBertForwardSAToSAPolicyDataset',
    'SocPreprocessedTextBertForwardSAToSADataset',
    'SocPreprocessedTextBertForwardSAToSAPolicyDataset',
]
# yapf: enable


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
