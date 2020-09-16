import os
from dataclasses import dataclass
import pandas as pd
import torch
from torch import Tensor
from transformers import BertModel, BertTokenizer
from typing import List, Optional, Dict
from .soc_psql import PSQLConfig
from .soc_psql_seq import SocPSQLSeqDataset
from . import utils as ds_utils
from .. import utils
from . import soc_data
# from ..typing import SocDatasetItem


@dataclass
class PSQLTextConfig(PSQLConfig):
    tokenizer_path: Optional[str] = None
    bert_model_path: Optional[str] = None
    use_pooler_features: bool = False
    set_empty_text_to_zero: bool = False


class SocPSQLTextBertSeqDataset(SocPSQLSeqDataset):
    """
        Defines a Settlers of Catan postgresql dataset for sequence models.
        One datapoint is a tuple (states, actions):
        - states is the full sequence of game states
        - actions is the full sequence of actions

        Args:
            psql_username: (str) username
            psql_host: (str) host
            psql_port: (int) port
            psql_db_name: (str) database name

        Returns:
            dataset: (Dataset) A pytorch Dataset giving access to the data

    """
    def _set_props(self, config):
        self.use_pooler_features = config['use_pooler_features']
        self.set_empty_text_to_zero = config['set_empty_text_to_zero']

        if config['tokenizer_path'] is not None:
            self.tokenizer = BertTokenizer.from_pretrained(config['tokenizer_path'])
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        if config['bert_model_path'] is not None:
            self.bert = BertModel.from_pretrained(config['bert_model_path'])
        else:
            self.bert = BertModel.from_pretrained('bert-base-cased')

        state_shape = [soc_data.STATE_SIZE] + soc_data.BOARD_SIZE
        action_shape = [soc_data.ACTION_SIZE] + soc_data.BOARD_SIZE
        if self.use_pooler_features:
            chat_shape = [self.bert.pooler.dense.out_features]
        else:
            chat_shape = [self.bert.encoder.layer[-1].output.dense.out_features]
        self.input_shape = [state_shape, action_shape, chat_shape]
        self.output_shape = [state_shape, action_shape, chat_shape]

        self.infix = 'text_bert'

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
            Return one datapoint from the dataset

            A datapoint is a complete trajectory (s_t, a_t, s_t+1, etc.)

        """
        state_seq_t, action_seq_t = super(SocPSQLTextBertSeqDataset, self).__getitem__(idx)
        game_length = state_seq_t.shape[0]

        chats_df = self._get_chats_from_db(idx)
        chats_df = ds_utils.preprocess_chats(chats_df, game_length)

        assert game_length == len(chats_df)

        messages = list(map(ds_utils.replace_firstnames, chats_df['message'].tolist()))
        with torch.no_grad():
            last_hidden_state, pooler_output, chat_mask_seq_t = ds_utils.compute_text_features(
                messages, self.tokenizer, self.bert
            )
        if self.use_pooler_features:
            chat_seq_t = pooler_output
        else:
            # last_hidden_state contains all the contextualized words for the padded sentence
            chat_seq_t = last_hidden_state

        data_dict = {
            'state_seq_t': state_seq_t,
            'action_seq_t': action_seq_t,
            'chat_seq_t': chat_seq_t,
            'chat_mask_seq_t': chat_mask_seq_t,
        }

        return data_dict

    def _get_chats_from_db(self, idx: int) -> pd.DataFrame:
        db_id = self._first_index + idx
        query = """
            SELECT *
            FROM chats_{}
        """.format(db_id)

        if self.engine is not None:
            with self.engine.connect() as conn:
                chats_df = pd.read_sql_query(query, con=conn)
        else:
            raise Exception('No engine detected')

        return chats_df

    def _load_input_seq(self, idx: int) -> List[Tensor]:
        data = self[idx]

        state_seq_t = data['state_seq_t']  # SxC_sxHxW
        action_seq_t = data['action_seq_t']  # SxC_axHxW
        chat_seq_t = data['chat_seq_t']  # SxF_c
        chat_mask_seq_t = data['chat_mask_seq_t']  # SxF_c

        input_seq_t = [torch.cat([state_seq_t, action_seq_t], dim=1), chat_seq_t, chat_mask_seq_t]

        return input_seq_t

    def _load_input_df_list(self, idx: int, testing: bool = False) -> List[pd.DataFrame]:
        df_list = super(SocPSQLTextBertSeqDataset, self)._load_input_df_list(idx, testing)

        chats_df = self._get_chats_from_db(idx)

        if testing is True:
            chats_df = chats_df[(chats_df['current_state'] >= df_list[0]['id'].min())
                                & (chats_df['current_state'] <= df_list[0]['id'].max())]

        df_list.append(chats_df)

        return df_list

    def save_assets(self, folder: str):
        utils.check_folder(folder)

        self.tokenizer.save_pretrained(os.path.join(folder, 'soc_tokenizer'))
        self.bert.save_pretrained(os.path.join(folder, 'soc_bert_model'))
