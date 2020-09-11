import os
import torch
from transformers import BertModel, BertTokenizer
from dataclasses import dataclass
from omegaconf import MISSING
from typing import Tuple, List, Union, Dict, Optional
from .soc_file_seq import SocFileSeqDataset
from . import utils as ds_utils
from .. import utils
from . import soc_data

SOCShape = Union[Tuple[List[int], ...], List[int]]


@dataclass
class FileTextConfig:
    name: str = MISSING
    tokenizer_path: Optional[str] = None
    bert_model_path: Optional[str] = None
    use_pooler_features: bool = False

    shuffle: bool = True


class SocFileTextBertSeqDataset(SocFileSeqDataset):
    """
        Defines a Settlers of Catan postgresql dataset for forward models.
        One datapoint is a tuple (past, future)

        Args:
            config: (Dict) The dataset configuration

        Returns:
            dataset: (Dataset) A pytorch Dataset giving access to the data

    """

    def _set_props(self, config):
        self.use_pooler_features = config['use_pooler_features']

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
        states_df, actions_df, chats_df = self._get_data(idx)

        game_length = len(states_df)
        assert game_length == len(states_df.index) == len(actions_df.index)

        states_df = ds_utils.preprocess_states(states_df)
        actions_df = ds_utils.preprocess_actions(actions_df)
        chats_df = ds_utils.preprocess_chats(chats_df, game_length)

        state_seq_t = ds_utils.stack_states_df(states_df)
        action_seq_t = ds_utils.stack_actions_df(actions_df)

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

    def _load_input_seq(self, idx: int) -> List[torch.Tensor]:
        data = self[idx]

        state_seq_t = data['state_seq_t']  # SxC_sxHxW
        action_seq_t = data['action_seq_t']  # SxC_axHxW
        chat_seq_t = data['chat_seq_t']  # SxF_c
        chat_mask_seq_t = data['chat_mask_seq_t']  # SxF_c

        input_seq_t = [torch.cat([state_seq_t, action_seq_t], dim=1), chat_seq_t, chat_mask_seq_t]

        return input_seq_t

    def save_assets(self, folder: str):
        utils.check_folder(folder)

        self.tokenizer.save_pretrained(os.path.join(folder, 'soc_tokenizer'))
        self.bert.save_pretrained(os.path.join(folder, 'soc_bert_model'))
