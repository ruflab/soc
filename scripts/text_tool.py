# flake8: noqa
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

# ckpt_path = os.path.join(cfd, 'results/resnet18bilstmconcat/humantrade.ckpt')
# ckpt_path = os.path.join(cfd, 'results/resnet18bilstmff/humantrade.ckpt')
# ckpt_path = os.path.join(cfd, 'results/resnet18bilstmffres/humantrade.ckpt')
# ckpt_path = os.path.join(cfd, 'results/resnet18bilstmffres/trade.ckpt')
ckpt_path = os.path.join(cfd, 'results/resnet18bilstmffres/full.ckpt')
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

ckpt['hyper_parameters']['dataset']['dataset_path'] = _RAW_SOC5_TEXT_BERT_DATASET_PATH
ckpt['hyper_parameters']['val_dataset']['dataset_path'] = _RAW_SOC5_TEXT_BERT_DATASET_PATH
ckpt['hyper_parameters']['dataset']['shuffle'] = False
ckpt['hyper_parameters']['batch_size'] = 1

pl.seed_everything(ckpt['hyper_parameters']['seed'])

runner = make_runner(ckpt['hyper_parameters'])
runner.setup('fit')
runner.load_state_dict(ckpt['state_dict'])
runner.eval()

pprint.pprint(runner.hparams)


def format_res_tensor(res_tensor):
    return res_tensor.view(4, 6).detach().numpy().tolist()


def format_res_pred(res_pred):
    return format_res_tensor(ds_utils.unnormalize_playersresources(res_pred))


def get_formated_preds(runner, output_pr_idx, x_seq, messages):
    train_dataset = runner.train_dataset
    last_hidden_state, pooler_output, chat_mask_seq_t = ds_utils.compute_text_features(
        messages, train_dataset.tokenizer, train_dataset.bert, train_dataset.set_empty_text_to_zero,
    )
    if train_dataset.use_pooler_features:
        chat_seq_t = pooler_output
    else:
        chat_seq_t = last_hidden_state
    x_text_seq = chat_seq_t[:train_dataset.history_length].unsqueeze(0)
    x_text_mask = chat_mask_seq_t[:train_dataset.history_length].unsqueeze(0)

    _, y_s_logits_seq, _ = runner.model(x_seq, x_text_seq, x_text_mask)
    res_pred = y_s_logits_seq[:, 0, output_pr_idx[0]:output_pr_idx[1]]

    return format_res_tensor(ds_utils.unnormalize_playersresources(res_pred))


def nice_print(t):
    print('           C, O, S,Wh,Wo,Unk')
    print('Betty:   ', t[0])
    print('Peter:   ', t[1])
    print('Jake:    ', t[2])
    print('Sam:     ', t[3])

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

        players_res = x_seq[:, -1, input_pr_idx[0]:input_pr_idx[1], 0, 0]
        players_res_true = y_s_true_seq[:, 0, output_pr_idx[0]:output_pr_idx[1]]
        players_res_preds = y_s_logits_seq[:, 0, output_pr_idx[0]:output_pr_idx[1]]

        _, y_s_logits_seq, _ = runner.model(x_seq, torch.zeros_like(x_text_seq), x_text_mask)
        players_res_preds_no_text = y_s_logits_seq[:, 0, output_pr_idx[0]:output_pr_idx[1]]

        _, y_s_logits_seq, _ = runner.model(torch.zeros_like(x_seq), x_text_seq, x_text_mask)
        players_res_preds_no_state = y_s_logits_seq[:, 0, output_pr_idx[0]:output_pr_idx[1]]

        print('\n')
        print(' -- Chat messages between states -- ')
        print(messages[0])
        print(' --\n')

        print(
            'Data: 4 players containing 6 resources in the following order CLAY ORE SHEEP WHEAT WOOD UNKNOWN'
        )
        print(
            'players:                                          -       Betty       -       Peter       -       Jake        -       Sam'
        )
        print('players_res (s^res_t)                            ', format_res_pred(players_res))

        print('players_res_true (s^res_t+1)                     ', format_res_pred(players_res_true))
        print('players_res_preds p(s^res_t+1|s_t, a_t, text_t)  ', format_res_pred(players_res_preds))
        print('players_res_preds p(s^res_t+1|s_t, a_t, 0)       ', format_res_pred(players_res_preds_no_text))
        print('players_res_preds p(s^res_t+1|0, 0, text_t)      ', format_res_pred(players_res_preds_no_state)
        )
        # print(
        #     'Distance predictions <-> true values:\n',
        #     (players_res_preds - players_res_true).view(4, -1)
        # )
        breakpoint()
