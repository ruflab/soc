import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from typing import List
from .soc_psql_seq import SocPSQLSeqDataset
from . import utils as ds_utils
# from ..typing import SocDatasetItem


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
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.tokenizer.add_tokens(['BayesBetty', 'BayesFranck', 'BayesJake', 'DRLSam'])
        self.tokenizer.add_tokens(['<void>'], special_tokens=True)
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.bert.resize_token_embeddings(len(self.tokenizer))

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
        chat_seq_t = self.bert(**encoded_inputs)

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

    def _load_input_seq(self, idx: int):
        data = self[idx]

        state_seq_t = data[0]  # SxC_sxHxW
        action_seq_t = data[1]  # SxC_axHxW
        chat_seq_t = data[2]  # SxC_cxHxW

        input_seq_t = torch.cat([state_seq_t, action_seq_t, chat_seq_t], dim=1)

        return input_seq_t

    def _load_input_df_list(self, idx: int, testing: bool = False) -> List:
        states_df = self._get_states_from_db(idx)
        actions_df = self._get_actions_from_db(idx)
        chats_df = self._get_chats_from_db(idx)

        if testing is True:
            chats_gb = chats_df.groupby('current_state')
            key = list(chats_gb.indices.keys())[0]
            chats_df = chats_gb.get_group(key).copy()
            chats_df['current_state'] = 5
            df_list = [states_df[:8], actions_df[:8], chats_df[:8]]
        else:
            df_list = [states_df, actions_df, chats_df]

        return df_list
