import os
import torch
from soc.datasets import soc_data

cfd = os.path.dirname(os.path.realpath(__file__))

_DATA_FOLDER = os.path.join(cfd, '..', 'data')
_RAW_SOC1_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_1_raw_df.pt')
_RAW_SOC5_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_5_raw_df.pt')
_RAW_SOC20_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_20_raw_df.pt')
_RAW_SOC50_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_50_raw_df.pt')
_RAW_SOC100_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_100_raw_df.pt')
_RAW_SOC500_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_500_raw_df.pt')
_RAW_SOC1000_TEXT_BERT_DATASET_PATH = os.path.join(_DATA_FOLDER, 'soc_text_bert_1000_raw_df.pt')

data = torch.load(_RAW_SOC50_TEXT_BERT_DATASET_PATH)

n_datapoints = 0
n_resources_changing = 0
n_trade = 0
n_fail_trade = 0
n_human_trade = 0
n_shop_trade = 0
n_bank_trade = 0

for states_df, actions_df, chats_df in data:
    for i in range(len(states_df) - 1):
        current_states_df = states_df[i:i + 2]

        min_id = current_states_df['id'].min()
        max_id = current_states_df['id'].min()

        current_actions_df = actions_df[(actions_df['beforestate'] >= min_id)
                                        & (actions_df['beforestate'] < max_id + 1)]

        # current_chats_df = chats_df[(chats_df['current_state'] >= min_id)
        #                             & (chats_df['current_state'] < max_id + 1)]

        n_datapoints += 1

        resources_df = current_states_df['playersresources']
        past_res = torch.tensor(resources_df.iloc[0])
        future_res = torch.tensor(resources_df.iloc[1])
        if not torch.equal(past_res, future_res):
            n_resources_changing += 1

        if current_actions_df['type'].iloc[0] == soc_data.ACTIONS['TRADE']:
            n_trade += 1

            if torch.equal(past_res, future_res):
                n_fail_trade += 1

            diff = torch.sum(future_res - past_res)
            if diff == 0:
                n_human_trade += 1
            elif diff == -3:
                n_bank_trade += 1
            else:
                n_shop_trade += 1

print('n_datapoints', n_datapoints)
print('n_resources_changing', n_resources_changing)
print('n_trade', n_trade)
print('n_fail_trade', n_fail_trade)
print('n_human_trade', n_human_trade)
print('n_bank_trade', n_bank_trade)
print('n_shop_trade', n_shop_trade)
