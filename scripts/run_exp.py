import os
import hydra
from omegaconf import DictConfig
from os.path import expanduser
import torch
from soc.training import train

cuda = torch.cuda.is_available()

cfd = os.path.dirname(os.path.realpath(__file__))
_DATA_FOLDER = os.path.join(cfd, '..', 'data')
_RAW_SOC1_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_1_raw_df.pt')
_RAW_SOC5_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_5_raw_df.pt')
_RAW_SOC20_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_20_raw_df.pt')
_RAW_SOC100_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_100_raw_df.pt')
_RAW_SOC500_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_500_raw_df.pt')
_RAW_SOC1000_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_1000_raw_df.pt')

if 'NEPTUNE_API_TOKEN' not in os.environ:
    home = expanduser("~")
    os.environ['NEPTUNE_API_TOKEN'] = open(home + "/.neptune.txt", "r").read().rstrip()
os.environ['NEPTUNE_PROJECT_NAME'] = 'morgangiraud/soc'

config_path = os.path.join("..", "experiments")
config_name = "005_gpu_resnet18concat_policy_full"


@hydra.main(config_path=config_path, config_name=config_name)
def main(config: DictConfig):

    # print('!!!Overrifing datasets!!!')
    # config.runner.dataset.dataset_path = _RAW_SOC5_TEXT_BERT_DATASET_PATH
    # config.runner.val_dataset.dataset_path = _RAW_SOC5_TEXT_BERT_DATASET_PATH

    _ = train(config)


if __name__ == "__main__":
    main()
