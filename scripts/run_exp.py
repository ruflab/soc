import os
from os.path import expanduser
import torch
from hydra.experimental import initialize, compose
from soc.training import train

cuda = torch.cuda.is_available()

cfd = os.path.dirname(os.path.realpath(__file__))
_DATA_FOLDER = os.path.join(cfd, '..', 'data')
_SOC10_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_10_fullseq.pt')
_SOC10_FOLDER_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_10_fullseq')
_SOC10_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_10_fullseq.pt')
_SOC50_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_50_fullseq.pt')
_SOC50_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_50_fullseq.pt')
_SOC150_FOLDER_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_150_fullseq')
_SOC150_RAW_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_150_raw.pt')

if 'NEPTUNE_API_TOKEN' not in os.environ:
    home = expanduser("~")
    os.environ['NEPTUNE_API_TOKEN'] = open(home + "/.neptune.txt", "r").read().rstrip()
os.environ['NEPTUNE_PROJECT_NAME'] = 'morgangiraud/soc'

# config_name = "002_gpu_resnet18_policy_overfit_reg_adamw"
# config_name = "003_gpu_convlstm_seq_policy_overfit"
# config_name = "004_gpu_resnet18_policy_4step"
config_name = "005_gpu_resetn18fusion_policy_overfit"
with initialize(config_path=os.path.join("..", "experiments")):
    config = compose(config_name=config_name)

    # config.generic.dataset.name = 'SocPSQLForwardSAToSAPolicyDataset'
    # config.generic.dataset.no_db = False
    # config.generic.dataset.psql_username = 'deepsoc'
    # config.generic.dataset.psql_host = 'localhost'
    # config.generic.dataset.psql_port = 5432
    # config.generic.dataset.psql_db_name = 'soc'
    # config.generic.dataset.first_index = 100
    # config.generic.dataset.name = 'SocLazyPreprocessedForwardSAToSAPolicyDataset'
    # config.generic.dataset.dataset_path = _SOC10_FOLDER_DATASET_PATH
    config.generic.dataset.dataset_path = _SOC10_TEXT_BERT_DATASET_PATH
    # config.generic.dataset.name = 'SocFileForwardSAToSAPolicyDataset'
    # config.generic.dataset.dataset_path = _SOC150_RAW_DATASET_PATH
    # config.generic.dataset.dataset_path = _SOC50_DATASET_PATH

    # config.generic.val_dataset.dataset_path = _SOC10_DATASET_PATH

    if cuda is False:
        config.trainer.gpus = 0

    print(config)

    runner = train(config)
