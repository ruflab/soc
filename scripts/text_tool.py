import os
import pprint
import torch
import pytorch_lightning as pl
import seaborn as sns
from omegaconf import DictConfig
# import matplotlib.pyplot as plt
from soc.runners import make_runner
# from soc.datasets import soc_data
import soc.datasets.utils as ds_utils
# from soc.val import compute_accs
# from soc.losses import compute_losses

sns.set(color_codes=True)

cfd = os.path.dirname(os.path.realpath(__file__))
_DATA_FOLDER = os.path.join(cfd, '..', 'data')
_RAW_SOC1_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_1_raw_df.pt')
_RAW_SOC5_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_5_raw_df.pt')
_RAW_SOC20_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_20_raw_df.pt')
_RAW_SOC100_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_100_raw_df.pt')
_RAW_SOC1000_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_1000_raw_df.pt')

# ckpt_path = os.path.join(cfd, 'results/resnet18meanconcat/last.ckpt')
# ckpt_path = os.path.join(cfd, 'results/resnet18meanff/last.ckpt')
ckpt_path = os.path.join(cfd, 'results/_ckpt_epoch_42.ckpt')
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
ckpt['hyper_parameters'] = DictConfig(ckpt['hyper_parameters'])
ckpt['hyper_parameters']['dataset'] = DictConfig(ckpt['hyper_parameters']['dataset'])
ckpt['hyper_parameters']['val_dataset'] = DictConfig(ckpt['hyper_parameters']['val_dataset'])
ckpt['hyper_parameters']['val_dataset']['name'] = ckpt['hyper_parameters']['dataset']['name']
ckpt['hyper_parameters']['val_dataset']['history_length'] = ckpt['hyper_parameters']['dataset'][
    'history_length']
ckpt['hyper_parameters']['val_dataset']['future_length'] = ckpt['hyper_parameters']['dataset'][
    'future_length']
ckpt['hyper_parameters']['val_dataset']['use_pooler_features'] = ckpt['hyper_parameters'][
    'dataset']['use_pooler_features']
ckpt['hyper_parameters']['val_dataset']['set_empty_text_to_zero'] = ckpt['hyper_parameters'][
    'dataset']['set_empty_text_to_zero']
ckpt['hyper_parameters']['model'] = DictConfig(ckpt['hyper_parameters']['model'])

ckpt['hyper_parameters']['dataset']['dataset_path'] = _RAW_SOC20_TEXT_BERT_DATASET_PATH
ckpt['hyper_parameters']['val_dataset']['dataset_path'] = _RAW_SOC20_TEXT_BERT_DATASET_PATH
ckpt['hyper_parameters']['dataset']['shuffle'] = False
ckpt['hyper_parameters']['batch_size'] = 1

pl.seed_everything(ckpt['hyper_parameters']['seed'])

runner = make_runner(ckpt['hyper_parameters'])
runner.setup('fit')
runner.load_state_dict(ckpt['state_dict'])
runner.eval()

pprint.pprint(runner.hparams)

input_meta = runner.train_dataset.get_input_metadata()
spatial_metadata, linear_metadata, actions_metadata = runner.output_metadata
input_pr_idx = input_meta['playersresources']
output_pr_idx = linear_metadata['playersresources']
with torch.no_grad():
    for idx, batch in enumerate(runner.train_dataloader()):
        x_seq = batch['history_t']
        x_text_seq = batch['chat_history_t']
        x_text_mask = batch['chat_mask_history_t']

        y_spatial_s_true_seq = batch['spatial_states_future_t']
        y_s_true_seq = batch['lin_states_future_t']
        y_a_true_seq = batch['actions_future_t']

        if torch.sum(x_text_mask) == 0:
            continue

        is_trade = ds_utils.find_actions_idxs(x_seq, 'TRADE')
        if is_trade is False:
            continue

        outputs = runner.model(x_seq, x_text_seq, x_text_mask)
        y_spatial_s_logits_seq, y_s_logits_seq, y_a_logits_seq = outputs

        full_seq_len = runner.train_dataset.history_length + runner.train_dataset.future_length
        states_df, actions_df, chats_df = runner.train_dataset._get_data(idx)
        p_chats_df = ds_utils.preprocess_chats(chats_df, full_seq_len, states_df['id'].min())
        messages = list(map(ds_utils.replace_firstnames, p_chats_df['message'].tolist()))

        players_resources = x_seq[:, -1, input_pr_idx[0]:input_pr_idx[1], 0, 0]
        players_resources_true = y_s_true_seq[:, 0, output_pr_idx[0]:output_pr_idx[1]]
        players_resources_preds = y_s_logits_seq[:, 0, output_pr_idx[0]:output_pr_idx[1]]
        print('Order: - 0, CLAY ORE SHEEP WHEAT WOOD UNKNOWN')
        print('players_resources        ', ds_utils.unnormalize_playersresources(players_resources))
        print(
            'players_resources_true   ',
            ds_utils.unnormalize_playersresources(players_resources_true)
        )
        print(
            'players_resources_preds  ',
            ds_utils.unnormalize_playersresources(players_resources_preds)
        )
        print(messages[0])
        print('raw_dist ', players_resources_preds - players_resources_true)
        breakpoint()
