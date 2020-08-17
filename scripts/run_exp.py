import os
from hydra.experimental import initialize, compose
from soc.training import train

cfd = os.path.dirname(os.path.realpath(__file__))
_DATA_FOLDER = os.path.join(cfd, '..', 'data')
_SOC50_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_50_fullseq.pt')
_SOC10_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_10_fullseq.pt')

os.environ['NEPTUNE_PROJECT_NAME'] = 'morgangiraud/soc'

# config_name = "002_gpu_resnet18_policy_overfit_reg_adamw"
config_name = "004_gpu_resnet18_policy"
with initialize(config_path=os.path.join("..", "experiments")):
    config = compose(config_name=config_name)

    config.generic.dataset.dataset_path = _SOC50_DATASET_PATH
    config.generic.val_dataset.dataset_path = _SOC10_DATASET_PATH
    # config.generic.batch_size = 4
    config.trainer.gpus = 0
    # config.other.save_top_k = 1
    # config.other.period = 499

    print(config)

    runner = train(config)
