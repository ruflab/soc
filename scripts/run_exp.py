import os
from hydra.experimental import initialize, compose
from soc.training import train

cfd = os.path.dirname(os.path.realpath(__file__))
_DATA_FOLDER = os.path.join(cfd, '..', 'data')
_SOC50_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_50_fullseq.pt')
_SOC150_FILE_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_150_fullseq')
_SOC150_RAW_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_150_raw.pt')
_SOC10_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_10_fullseq.pt')

os.environ['NEPTUNE_PROJECT_NAME'] = 'morgangiraud/soc'

# config_name = "002_gpu_resnet18_policy_overfit_reg_adamw"
config_name = "004_gpu_resnet18_policy"
# config_name = "003_gpu_convlstm_seq_policy_overfit"
with initialize(config_path=os.path.join("..", "experiments")):
    config = compose(config_name=config_name)

    # config.generic.dataset.name = 'SocFilePreprocessedForwardSAToSAPolicyDataset'
    # config.generic.dataset.dataset_path = _SOC150_FILE_DATASET_PATH
    config.generic.dataset.name = 'SocFileForwardSAToSAPolicyDataset'
    config.generic.dataset.dataset_path = _SOC150_RAW_DATASET_PATH
    # config.generic.dataset.dataset_path = _SOC50_DATASET_PATH

    config.generic.val_dataset.dataset_path = _SOC10_DATASET_PATH

    config.trainer.gpus = 0

    print(config)

    runner = train(config)
