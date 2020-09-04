import os
from dataclasses import dataclass
import pandas as pd
import torch
from torch import Tensor
from transformers import BertModel, BertTokenizer
from typing import List, Optional
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
    use_pooler_features: bool = True


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

        if config['tokenizer_path'] is not None:
            self.tokenizer = BertTokenizer.from_pretrained(config['tokenizer_path'])
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.tokenizer.add_tokens(['BayesBetty', 'BayesFranck', 'BayesJake', 'DRLSam'])
            self.tokenizer.add_tokens(['<void>'], special_tokens=True)

        if config['bert_model_path'] is not None:
            self.bert = BertModel.from_pretrained(config['bert_model_path'])
        else:
            self.bert = BertModel.from_pretrained('bert-base-cased')
            self.bert.resize_token_embeddings(len(self.tokenizer))

        state_shape = [soc_data.STATE_SIZE] + soc_data.BOARD_SIZE
        action_shape = [soc_data.ACTION_SIZE] + soc_data.BOARD_SIZE
        if self.use_pooler_features:
            chat_shape = [self.bert.pooler.dense.out_features]
        else:
            chat_shape = [self.bert.encoder.layer[-1].output.dense.out_features]
        self.input_shape = [state_shape, action_shape, chat_shape]
        self.output_shape = [state_shape, action_shape, chat_shape]

        self.infix = 'text_bert'

    def __getitem__(self, idx: int):
        """
            Return one datapoint from the dataset

            A datapoint is a complete trajectory (s_t, a_t, s_t+1, etc.)

        """
        state_seq_t, action_seq_t = super(SocPSQLTextBertSeqDataset, self).__getitem__(idx)
        game_length = state_seq_t.shape[0]

        chats_df = self._get_chats_from_db(idx)
        chats_df = ds_utils.preprocess_chats(chats_df, game_length)

        assert game_length == len(chats_df)

        encoded_inputs = self.tokenizer(
            chats_df['message'].tolist(), padding=True, truncation=True, return_tensors="pt"
        )

        # For now, we will use the pooler_output, but HuggingFace advise against it
        # https://huggingface.co/transformers/model_doc/bert.html
        with torch.no_grad():
            # I have to check if the no_grad call does not create problems with pytorch_lightning
            last_hidden_state, pooler_output = self.bert(**encoded_inputs)
        if self.use_pooler_features:
            chat_seq_t = pooler_output
        else:
            raise NotImplementedError('Using all Bert hidden states is not implemented yet')

        return state_seq_t, action_seq_t, chat_seq_t

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

        state_seq_t = data[0]  # SxC_sxHxW
        action_seq_t = data[1]  # SxC_axHxW
        chat_seq_t = data[2]  # SxF_c

        input_seq_t = [torch.cat([state_seq_t, action_seq_t], dim=1), chat_seq_t]

        return input_seq_t

    def _load_input_df_list(self, idx: int, testing: bool = False) -> List[pd.DataFrame]:
        states_df = self._get_states_from_db(idx)
        actions_df = self._get_actions_from_db(idx)
        chats_df = self._get_chats_from_db(idx)

        if testing is True:
            chats_gb = chats_df.groupby('current_state')
            key = list(chats_gb.indices.keys())[0]
            chats_df = chats_gb.get_group(key).copy()
            chats_df['current_state'] = 5
            sec_trunc_idx = 20
            df_list = [
                states_df[10:10 + sec_trunc_idx],
                actions_df[10:10 + sec_trunc_idx],
                chats_df[10:10 + sec_trunc_idx],
            ]
        else:
            df_list = [states_df, actions_df, chats_df]

        return df_list

    def save_assets(self, folder: str):
        utils.check_folder(folder)

        self.tokenizer.save_pretrained(os.path.join(folder, 'soc_tokenizer'))
        self.bert.save_pretrained(os.path.join(folder, 'soc_bert_model'))
