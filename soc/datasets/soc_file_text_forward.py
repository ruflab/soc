from dataclasses import dataclass
import torch
from omegaconf import MISSING
from transformers import BertModel, BertTokenizer
from typing import List, Union, Tuple, Dict
from .soc_file_text_seq import FileTextConfig, SocFileTextBertSeqDataset
from . import utils as ds_utils
from . import soc_data
from ..typing import SocDataMetadata


@dataclass
class FileTextForwardConfig(FileTextConfig):
    history_length: int = MISSING
    future_length: int = MISSING


class SocFileTextBertForwardSAToSADataset(SocFileTextBertSeqDataset):
    """
        Defines a Settlers of Catan postgresql dataset for forward models.
        One datapoint is a tuple (past, future)

        Args:
            config: (Dict) The dataset configuration

        Returns:
            dataset: (Dataset) A pytorch Dataset giving access to the data

    """
    def _set_props(self, config):
        self.history_length = config['history_length']
        self.future_length = config['future_length']
        self.seq_len_per_datum = self.history_length + self.future_length
        self.use_pooler_features = config['use_pooler_features']
        self.set_empty_text_to_zero = config['set_empty_text_to_zero']
        self._inc_seq_steps: List[int] = []
        self._length = -1
        self.use_gpu = False

        if config['tokenizer_path'] is not None:
            self.tokenizer = BertTokenizer.from_pretrained(config['tokenizer_path'])
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        if config['bert_model_path'] is not None:
            self.bert = BertModel.from_pretrained(config['bert_model_path'])
        else:
            self.bert = BertModel.from_pretrained('bert-base-cased')
        if torch.cuda.is_available():
            self.bert = self.bert.cuda()
            self.use_gpu = True

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

    def __len__(self) -> int:
        return self._get_length()

    def _get_length(self) -> int:
        if self._length == -1:
            total_steps = 0
            nb_games = len(self.data)
            for i in range(nb_games):
                seq = self.data[i]
                if isinstance(seq, list):
                    seq = seq[0]
                total_steps += seq.shape[0]

            self._length = total_steps - nb_games * (self.seq_len_per_datum - 1)

        return self._length

    def _set_stats(self):
        nb_steps = self._get_nb_steps()
        for i, nb_step in enumerate(nb_steps):
            seq_nb_steps = nb_step - (self.seq_len_per_datum - 1)

            if i == 0:
                self._inc_seq_steps.append(seq_nb_steps)
            else:
                self._inc_seq_steps.append(seq_nb_steps + self._inc_seq_steps[-1])

    def _get_nb_steps(self) -> List[int]:
        nb_games = len(self.data)
        nb_steps = []
        for i in range(nb_games):
            seq = self.data[i]
            if isinstance(seq, list):
                seq = seq[0]
            nb_steps.append(seq.shape[0])

        return nb_steps

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        full_seq_len = self.history_length + self.future_length
        table_id, start_row_id, end_row_id = self._get_db_idxs(idx)

        states_df, actions_df, chats_df = self._get_data(idx)

        first_state_idx = start_row_id + 1

        states_df = ds_utils.preprocess_states(states_df)
        actions_df = ds_utils.preprocess_actions(actions_df)
        chats_df = ds_utils.preprocess_chats(chats_df, full_seq_len, first_state_idx)

        assert len(states_df.index) == len(actions_df.index) == len(chats_df.index)

        state_seq_t = ds_utils.stack_states_df(states_df)
        action_seq_t = ds_utils.stack_actions_df(actions_df)

        messages = list(map(ds_utils.replace_firstnames, chats_df['message'].tolist()))

        last_hidden_state, pooler_output, chat_mask_seq_t = ds_utils.compute_text_features(
            messages, self.tokenizer, self.bert, self.set_empty_text_to_zero,
        )
        if self.use_pooler_features:
            chat_seq_t = pooler_output
        else:
            # last_hidden_state contains all the contextualized words for the padded sentence
            chat_seq_t = last_hidden_state

        seq_t = torch.cat([state_seq_t, action_seq_t], dim=1)
        history_t = seq_t[:self.history_length]
        future_t = seq_t[self.history_length:]
        chat_history_t = chat_seq_t[:self.history_length]
        chat_future_t = chat_seq_t[self.history_length:]
        chat_mask_history_t = chat_mask_seq_t[:self.history_length]
        chat_mask_future_t = chat_mask_seq_t[self.history_length:]

        if self.use_gpu is True:
            new_data_dict = {
                'history_t': history_t.cuda(),
                'chat_history_t': chat_history_t.cuda(),
                'chat_mask_history_t': chat_mask_history_t.cuda(),
                'future_t': future_t.cuda(),
                'chat_future_t': chat_future_t.cuda(),
                'chat_mask_future_t': chat_mask_future_t.cuda(),
            }
        else:
            new_data_dict = {
                'history_t': history_t,
                'chat_history_t': chat_history_t,
                'chat_mask_history_t': chat_mask_history_t,
                'future_t': future_t,
                'chat_future_t': chat_future_t,
                'chat_mask_future_t': chat_mask_future_t,
            }

        return new_data_dict

    def _get_data(self, idx: int):
        table_id, start_row_id, end_row_id = self._get_db_idxs(idx)

        states_df, actions_df, chats_df = self.data[table_id]

        states_df = states_df[start_row_id:end_row_id]
        actions_df = actions_df[(actions_df['beforestate'] >= start_row_id + 1)
                                & (actions_df['beforestate'] < end_row_id + 1)]

        chats_df = chats_df[(chats_df['current_state'] >= start_row_id + 1)
                            & (chats_df['current_state'] < end_row_id + 1)]

        return states_df, actions_df, chats_df

    def _get_db_idxs(self, idx: int) -> Tuple:
        if len(self._inc_seq_steps) == 0:
            self._set_stats()

        prev_seq_steps = 0
        table_id = 0
        for i, seq_steps in enumerate(self._inc_seq_steps):
            if idx < seq_steps:
                table_id = i
                break
            prev_seq_steps = seq_steps
        r = idx - prev_seq_steps
        start_row_id = r  # We do not add one as dataframe starts at index 0
        end_row_id = start_row_id + self.seq_len_per_datum

        return table_id, start_row_id, end_row_id

    def get_output_metadata(self) -> Union[SocDataMetadata, Tuple[SocDataMetadata, ...]]:
        metadata: SocDataMetadata = {}
        last_idx = 0

        for field in soc_data.STATE_FIELDS:
            metadata['mean_' + field] = [last_idx, last_idx + soc_data.STATE_FIELDS_SIZE[field]]
            last_idx += soc_data.STATE_FIELDS_SIZE[field]

        metadata['mean_actions'] = [last_idx, last_idx + soc_data.ACTION_SIZE]

        return metadata

    def get_collate_fn(self):
        return ds_utils.pad_seq_text_policy


class SocFileTextBertForwardSAToSAPolicyDataset(SocFileTextBertForwardSAToSADataset):
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
        self.set_empty_text_to_zero = config['set_empty_text_to_zero']
        self._inc_seq_steps = []
        self._length = -1
        self.use_gpu = False

        if config['tokenizer_path'] is not None:
            self.tokenizer = BertTokenizer.from_pretrained(config['tokenizer_path'])
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        if config['bert_model_path'] is not None:
            self.bert = BertModel.from_pretrained(config['bert_model_path'])
        else:
            self.bert = BertModel.from_pretrained('bert-base-cased')
        if torch.cuda.is_available():
            self.bert = self.bert.cuda()
            self.use_gpu = True

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
        data_dict = super(SocFileTextBertForwardSAToSAPolicyDataset, self).__getitem__(idx)

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
