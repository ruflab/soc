from dataclasses import dataclass
import pandas as pd
import torch
from omegaconf import MISSING
from transformers import BertModel, BertTokenizer
from typing import List, Union, Tuple
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
            self.tokenizer.add_tokens(['BayesBetty', 'BayesFranck', 'BayesJake', 'DRLSam'])
            self.tokenizer.add_tokens(['<void>'], special_tokens=True)

        if config['bert_model_path'] is not None:
            self.bert = BertModel.from_pretrained(config['bert_model_path'])
        else:
            self.bert = BertModel.from_pretrained('bert-base-cased')
            self.bert.resize_token_embeddings(len(self.tokenizer))

        game_input_shape = [
            self.history_length, soc_data.STATE_SIZE + soc_data.ACTION_SIZE
        ] + soc_data.BOARD_SIZE
        if self.use_pooler_features:
            text_input_shape = [self.history_length, self.bert.pooler.dense.out_features]
        else:
            text_input_shape = [
                self.history_length, self.bert.encoder.layer[-1].output.dense.out_features
            ]
        self.input_shape = [game_input_shape, text_input_shape]

        self.output_shape = [
            self.future_length, soc_data.STATE_SIZE + soc_data.ACTION_SIZE
        ] + soc_data.BOARD_SIZE

    def __getitem__(self, idx: int):
        """
            Return one datapoint from the dataset

            A datapoint is a complete trajectory (s_t, a_t, s_t+1, etc.)

        """
        history_t, future_t = super(SocPSQLTextBertForwardSAToSADataset, self).__getitem__(idx)

        full_seq_len = history_t.shape[0] + future_t.shape[0]
        table_id, start_row_id, end_row_id = self._get_db_idxs(idx)
        chats_df = self._get_chats_from_db(table_id, start_row_id, end_row_id)

        chats_df = ds_utils.preprocess_chats(chats_df, full_seq_len, start_row_id)

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
        chat_history_t = chat_seq_t[:self.history_length]
        chat_future_t = chat_seq_t[self.history_length:]

        return [history_t, chat_history_t], [future_t, chat_future_t]

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
            self.tokenizer.add_tokens(['BayesBetty', 'BayesFranck', 'BayesJake', 'DRLSam'])
            self.tokenizer.add_tokens(['<void>'], special_tokens=True)

        if config['bert_model_path'] is not None:
            self.bert = BertModel.from_pretrained(config['bert_model_path'])
        else:
            self.bert = BertModel.from_pretrained('bert-base-cased')
            self.bert.resize_token_embeddings(len(self.tokenizer))

        game_input_shape = [
            self.history_length, soc_data.STATE_SIZE + soc_data.ACTION_SIZE
        ] + soc_data.BOARD_SIZE
        if self.use_pooler_features:
            text_input_shape = [self.history_length, self.bert.pooler.dense.out_features]
        else:
            text_input_shape = [
                self.history_length, self.bert.encoder.layer[-1].output.dense.out_features
            ]
        self.input_shape = [game_input_shape, text_input_shape]

        output_shape_spatial = [
            self.future_length, soc_data.SPATIAL_STATE_SIZE
        ] + soc_data.BOARD_SIZE
        output_shape = [self.future_length, soc_data.STATE_SIZE - soc_data.SPATIAL_STATE_SIZE]
        output_shape_actions = [self.future_length, soc_data.ACTION_SIZE]
        self.output_shape = (output_shape_spatial, output_shape, output_shape_actions)

    def __getitem__(self, idx: int):
        history_l, future_l = super(
            SocPSQLTextBertForwardSAToSAPolicyDataset, self
        ).__getitem__(idx)
        history_t = history_l[0]
        history_chat_t = history_l[1]
        future_t = future_l[0]
        future_chat_t = future_l[1]

        future_states_t = future_t[:, :-soc_data.ACTION_SIZE]  # [S, C_s, H, W]
        future_actions_t = future_t[:, -soc_data.ACTION_SIZE:, 0, 0]  # [S, C_a]
        future_spatial_states_t = torch.cat([future_states_t[:, 0:3], future_states_t[:, 9:81]],
                                            dim=1)  # [S, C_ss, H, W]
        future_lin_states_t = torch.cat(
            [future_states_t[:, 3:9, 0, 0], future_states_t[:, 81:, 0, 0]], dim=1
        )  # [S, C_ls]

        return ([history_t, history_chat_t],
                [future_spatial_states_t, future_lin_states_t, future_actions_t, future_chat_t])

    def get_output_metadata(self) -> Union[SocDataMetadata, Tuple[SocDataMetadata, ...]]:
        spatial_metadata: SocDataMetadata = {}
        last_spatial_idx = 0

        linear_metadata: SocDataMetadata = {}
        last_linear_idx = 0

        for field in soc_data.STATE_FIELDS:
            field_type = soc_data.STATE_FIELDS_TYPE[field]
            if field_type in [3, 4, 5]:
                spatial_metadata[field] = [
                    last_spatial_idx,
                    last_spatial_idx + soc_data.STATE_FIELDS_SIZE[field]
                ]
                last_spatial_idx += soc_data.STATE_FIELDS_SIZE[field]
            else:
                linear_metadata[field] = [
                    last_linear_idx,
                    last_linear_idx + soc_data.STATE_FIELDS_SIZE[field]
                ]
                last_linear_idx += soc_data.STATE_FIELDS_SIZE[field]

        actions_metadata: SocDataMetadata = {
            'actions': [0, soc_data.ACTION_SIZE],
        }

        return (spatial_metadata, linear_metadata, actions_metadata)
