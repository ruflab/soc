import os
# import hydra
from omegaconf import DictConfig
import torch
from soc.datasets import make_dataset

cuda = torch.cuda.is_available()

cfd = os.path.dirname(os.path.realpath(__file__))
_DATA_FOLDER = os.path.join(cfd, '..', 'data')
_SOC50_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_50_fullseq.pt')
_RAW_SOC1_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_1_raw_df.pt')
_RAW_SOC5_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_5_raw_df.pt')
_RAW_SOC20_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_20_raw_df.pt')
_RAW_SOC100_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_100_raw_df.pt')
_RAW_SOC500_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_500_raw_df.pt')
_RAW_SOC1000_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_1000_raw_df.pt')

config = {
    'name': 'SocFileTextBertHumanTradeForwardSAToSAPolicyDataset',
    'history_length': 1,
    'future_length': 1,
    'use_pooler_features': False,
    'set_empty_text_to_zero': True,
    'tokenizer_path': None,
    'bert_model_path': None,
    'shuffle': True,
    'dataset_path': _RAW_SOC500_TEXT_BERT_DATASET_PATH
}

dataset = make_dataset(DictConfig(config))
breakpoint()
