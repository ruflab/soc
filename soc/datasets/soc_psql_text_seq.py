import pandas as pd
import torch
# from typing import Tuple, List

from .soc_psql_seq import SocPSQLSeqDataset
from . import utils as ds_utils
from .. import utils
# from ..typing import SocDatasetItem


class SocPSQLTextSeqDataset(SocPSQLSeqDataset):
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

    def __getitem__(self, idx: int):
        """
            Return one datapoint from the dataset

            A datapoint is a complete trajectory (s_t, a_t, s_t+1, etc.)

        """
        state_seq_t, action_seq_t = super(SocPSQLTextSeqDataset, self).__getitem__(idx)
        game_length = state_seq_t.shape[0]

        chats_df = self._get_chats_from_db(idx)
        chats_df = ds_utils.preprocess_chats(chats_df, game_length)

        assert game_length == len(chats_df)

        # chat_seq = []
        # for i in range(game_length):
        #     current_chat_df = chats_df.iloc[i]
        #     current_chat_np = current_chat_df['message']

        #     chat_seq.append(torch.tensor(current_chat_np, dtype=torch.int64))
        # chat_seq_t = torch.stack(chat_seq)

        return (state_seq_t, action_seq_t), chats_df['message'].tolist()

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

    def dump_raw_dataset(self, folder: str):
        utils.check_folder(folder)

        limit = len(self)

        data = []
        for i in range(limit):
            states_df = self._get_states_from_db(i)
            actions_df = self._get_actions_from_db(i)
            chats_df = self._get_chats_from_db(i)

            data.append([states_df, actions_df, chats_df])

        path = "{}/soc_{}_raw.pt".format(folder, limit)
        torch.save(data, path)
