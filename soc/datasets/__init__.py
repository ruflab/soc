from .soc_psql_seq import SocPSQLSeqDataset, SocPSQLSeqSAToSDataset

__all__ = [
    'SocPSQLSeqDataset',
    'SocPSQLSeqSAToSDataset', ]


def make_dataset(config):
    if config['dataset'] in __all__:
        if 'no_db' in config:
            no_db = config['no_db']
        else:
            no_db = False
        return globals()[config['dataset']](no_db=no_db)
    else:
        raise Exception('The dataset name {} does not exist'.format(config['dataset']))


def get_dataset_class(config):
    if config['dataset'] in __all__:
        return globals()[config['dataset']]
    else:
        raise Exception('The dataset name {} does not exist'.format(config['dataset']))


def get_datasets_list():
    return __all__
