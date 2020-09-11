from dataclasses import dataclass
import pandas as pd
import torch
from omegaconf import MISSING
from transformers import BertModel, BertTokenizer
from typing import List, Union, Tuple, Dict
from .soc_psql_text_seq import PSQLTextConfig
from .soc_psql_forward import SocPSQLForwardSAToSADataset
from . import utils as ds_utils
from . import soc_data
from ..typing import SocDataMetadata


@dataclass
class PSQLTextForwardConfig(PSQLTextConfig):
    history_length: int = MISSING
    future_length: int = MISSING


class SocPSQLTextBertForwardSAToSADataset(SocPSQLForwardSAToSADataset):
    """
        Defines a Settlers of Catan postgresql dataset for forward models.
        One datapoint is a tuple (past, future)

        Args:
            config: (Dict) The dataset configuration

        Returns:
            dataset: (Dataset) A pytorch Dataset giving access to the data

    """

    _inc_seq_steps: List = []
    history_length: int
    future_length: int

    def _set_props(self, config):
        self.history_length = config['history_length']
        self.future_length = config['future_length']
        self.seq_len_per_datum = self.history_length + self.future_length
        self.use_pooler_features = config['use_pooler_features']

        if config['tokenizer_path'] is not None:
            self.tokenizer = BertTokenizer.from_pretrained(config['tokenizer_path'])
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        if config['bert_model_path'] is not None:
            self.bert = BertModel.from_pretrained(config['bert_model_path'])
        else:
            self.bert = BertModel.from_pretrained('bert-base-cased')

        game_input_shape = [
            self.history_length, soc_data.STATE_SIZE + soc_data.ACTION_SIZE
        ] + soc_data.BOARD_SIZE
        if self.use_pooler_features:
            text_input_shape = [self.history_length, None, self.bert.pooler.dense.out_features]
        else:
            text_input_shape = [
                self.history_length, None, self.bert.encoder.layer[-1].output.dense.out_features
            ]
        self.input_shape = [game_input_shape, text_input_shape]

        self.output_shape = [
            self.future_length, soc_data.STATE_SIZE + soc_data.ACTION_SIZE
        ] + soc_data.BOARD_SIZE

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
            Return one datapoint from the dataset

            A datapoint is a complete trajectory (s_t, a_t, s_t+1, etc.)

        """
        history_t, future_t = super(SocPSQLTextBertForwardSAToSADataset, self).__getitem__(idx)

        full_seq_len = history_t.shape[0] + future_t.shape[0]
        table_id, start_row_id, end_row_id = self._get_db_idxs(idx)
        chats_df = self._get_chats_from_db(table_id, start_row_id, end_row_id)

        chats_df = ds_utils.preprocess_chats(chats_df, full_seq_len, start_row_id)

        messages = list(map(ds_utils.replace_firstnames, chats_df['message'].tolist()))
        with torch.no_grad():
            last_hidden_state, pooler_output, mask = ds_utils.compute_text_features(
                messages, self.tokenizer, self.bert
            )
        if self.use_pooler_features:
            chat_seq_t = pooler_output
        else:
            # last_hidden_state contains all the contextualized words for the padded sentence
            chat_seq_t = last_hidden_state
        chat_history_t = chat_seq_t[:self.history_length]
        chat_mask_history_t = mask[:self.history_length]
        chat_future_t = chat_seq_t[self.history_length:]
        chat_mask_future_t = mask[self.history_length:]

        data_dict = {
            'history_t': history_t,
            'chat_history_t': chat_history_t,
            'chat_mask_history_t': chat_mask_history_t,
            'future_t': future_t,
            'chat_future_t': chat_future_t,
            'chat_mask_future_t': chat_mask_future_t,
        }

        return data_dict

    def _get_chats_from_db(self, table_id: int, start_row_id: int, end_row_id: int) -> pd.DataFrame:
        query = """
            SELECT *
            FROM chats_{}
            WHERE current_state >= {} AND current_state < {}
        """.format(table_id, start_row_id, end_row_id)

        if self.engine is not None:
            with self.engine.connect() as conn:
                chats_df = pd.read_sql_query(query, con=conn)
        else:
            raise Exception('No engine detected')

        return chats_df

    def get_collate_fn(self):
        return None


class SocPSQLTextBertForwardSAToSAPolicyDataset(SocPSQLTextBertForwardSAToSADataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: [S_h, (C_states + C_actions), H, W]

        Output: Tuple of next state and next actions
            Dims: ( [S_f, C_ss, H, W], [S_f, C_ls], [S_f, C_actions] )
    """
    def _set_props(self, config):
        self.history_length = config['history_length']
        self.future_length = config['future_length']
        self.seq_len_per_datum = self.history_length + self.future_length
        self.use_pooler_features = config['use_pooler_features']

        if config['tokenizer_path'] is not None:
            self.tokenizer = BertTokenizer.from_pretrained(config['tokenizer_path'])
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        if config['bert_model_path'] is not None:
            self.bert = BertModel.from_pretrained(config['bert_model_path'])
        else:
            self.bert = BertModel.from_pretrained('bert-base-cased')

        game_input_shape = [
            self.history_length, soc_data.STATE_SIZE + soc_data.ACTION_SIZE
        ] + soc_data.BOARD_SIZE
        if self.use_pooler_features:
            text_input_shape = [self.history_length, None, self.bert.pooler.dense.out_features]
        else:
            text_input_shape = [
                self.history_length, None, self.bert.encoder.layer[-1].output.dense.out_features
            ]
        self.input_shape = [game_input_shape, text_input_shape]

        output_shape_spatial = [
            self.future_length, soc_data.SPATIAL_STATE_SIZE
        ] + soc_data.BOARD_SIZE
        output_shape = [self.future_length, soc_data.STATE_SIZE - soc_data.SPATIAL_STATE_SIZE]
        output_shape_actions = [self.future_length, soc_data.ACTION_SIZE]
        self.output_shape = (output_shape_spatial, output_shape, output_shape_actions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_dict = super(SocPSQLTextBertForwardSAToSAPolicyDataset, self).__getitem__(idx)

        future_t = data_dict['future_t']
        del data_dict['future_t']

        states_future_t = future_t[:, :-soc_data.ACTION_SIZE]  # [S, C_s, H, W]
        actions_future_t = future_t[:, -soc_data.ACTION_SIZE:, 0, 0]  # [S, C_a]

        spatial_states_future_t, lin_states_future_t = ds_utils.separate_state_data(states_future_t)

        data_dict['spatial_states_future_t'] = spatial_states_future_t
        data_dict['lin_states_future_t'] = lin_states_future_t
        data_dict['actions_future_t'] = actions_future_t

        return data_dict

    def get_output_metadata(self) -> Union[SocDataMetadata, Tuple[SocDataMetadata, ...]]:
        spatial_metadata: SocDataMetadata = {}
        last_spatial_idx = 0

        linear_metadata: SocDataMetadata = {}
        last_linear_idx = 0

        for field in soc_data.STATE_FIELDS:
            field_type = soc_data.STATE_FIELDS_TYPE[field]
            if field_type in [3, 4, 5]:
                spatial_metadata[field] = [
                    last_spatial_idx, last_spatial_idx + soc_data.STATE_FIELDS_SIZE[field]
                ]
                last_spatial_idx += soc_data.STATE_FIELDS_SIZE[field]
            else:
                linear_metadata[field] = [
                    last_linear_idx, last_linear_idx + soc_data.STATE_FIELDS_SIZE[field]
                ]
                last_linear_idx += soc_data.STATE_FIELDS_SIZE[field]

        actions_metadata: SocDataMetadata = {
            'actions': [0, soc_data.ACTION_SIZE],
        }

        return (spatial_metadata, linear_metadata, actions_metadata)
