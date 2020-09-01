import os
import torch
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
from ..typing import SocDataMetadata
from . import soc_data
from .soc_preprocessed_forward import PreprocessedForwardConfig

cfd = os.path.dirname(os.path.realpath(__file__))
_DATA_FOLDER = os.path.join(cfd, '..', '..', 'data')

SOCShape = Union[Tuple[List[int], ...], List[int]]


@dataclass
class PreprocessedTextForwardConfig(PreprocessedForwardConfig):
    tokenizer_path: Optional[str] = None
    bert_model_path: Optional[str] = None
    use_pooler_features: bool = True


class SocPreprocessedTextBertForwardSAToSADataset(Dataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: [S, (C_states + C_actions), H, W]

        Output: Next state
            Dims: [S, (C_states + C_actions), H, W]
    """

    _length: int = -1
    _inc_seq_steps: List[int] = []
    history_length: int
    future_length: int
    input_shape: SOCShape
    output_shape: SOCShape

    def __init__(self, omegaConf, dataset_type: str = 'train'):
        super(SocPreprocessedTextBertForwardSAToSADataset, self).__init__()

        self.path = omegaConf['dataset_path']
        self.history_length = omegaConf['history_length']
        self.future_length = omegaConf['future_length']
        self.seq_len_per_datum = self.history_length + self.future_length
        self.use_pooler_features = omegaConf['use_pooler_features']

        if omegaConf['tokenizer_path'] is not None:
            self.tokenizer = BertTokenizer.from_pretrained(omegaConf['tokenizer_path'])
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.tokenizer.add_tokens(['BayesBetty', 'BayesFranck', 'BayesJake', 'DRLSam'])
            self.tokenizer.add_tokens(['<void>'], special_tokens=True)

        if omegaConf['bert_model_path'] is not None:
            self.bert = BertModel.from_pretrained(omegaConf['bert_model_path'])
        else:
            self.bert = BertModel.from_pretrained('bert-base-cased')
            self.bert.resize_token_embeddings(len(self.tokenizer))

        self.data = torch.load(self.path)

        self._set_props(omegaConf)

    def _set_props(self, omegaConf):
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

            self._length = total_steps - nb_games * self.seq_len_per_datum

        return self._length

    def _set_stats(self):
        nb_steps = self._get_nb_steps()
        for i, nb_step in enumerate(nb_steps):
            seq_nb_steps = nb_step - self.seq_len_per_datum

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

    def __getitem__(self, idx: int):
        x_t, x_text_t = self._get_data(idx)

        _, _, H, W = x_t.shape
        history_t = x_t[:self.history_length]
        future_t = x_t[self.history_length:]
        chat_history_t = x_text_t[:self.history_length]
        chat_future_t = x_text_t[self.history_length:]

        return [history_t, chat_history_t], [future_t, chat_future_t]

    def _get_data(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        start_row_id = r
        end_row_id = r + self.seq_len_per_datum

        game_seq_t = self.data[table_id][0][start_row_id:end_row_id]
        chat_seq_t = self.data[table_id][1][start_row_id:end_row_id]

        return game_seq_t, chat_seq_t

    def get_input_size(self) -> SOCShape:
        """
            Return the input dimension
        """

        return self.input_shape

    def get_output_size(self) -> SOCShape:
        """
            Return the output dimension
        """

        return self.output_shape

    def get_collate_fn(self) -> None:
        return None

    def get_output_metadata(self) -> SocDataMetadata:
        metadata: SocDataMetadata = {
            'hexlayout': [0, 1],
            'numberlayout': [1, 2],
            'mean_robberhex': [2, 3],
            'mean_piecesonboard': [3, 75],
            'mean_gamestate': [75, 99],
            'mean_diceresult': [99, 112],
            'mean_startingplayer': [112, 116],
            'mean_currentplayer': [116, 120],
            'devcardsleft': [120, 121],
            'mean_playeddevcard': [121, 122],
            'players': [122, 286],
            'mean_actions': [286, 303],
        }

        return metadata


class SocPreprocessedTextBertForwardSAToSAPolicyDataset(
    SocPreprocessedTextBertForwardSAToSADataset
):
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
            SocPreprocessedTextBertForwardSAToSAPolicyDataset, self
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

    def get_output_metadata(self):
        spatial_metadata: SocDataMetadata = {
            'hexlayout': [0, 1],
            'numberlayout': [1, 2],
            'robberhex': [2, 3],
            'piecesonboard': [3, 75],
        }

        linear_metadata: SocDataMetadata = {
            'gamestate': [0, 24],
            'diceresult': [24, 37],
            'startingplayer': [37, 41],
            'currentplayer': [41, 45],
            'devcardsleft': [45, 46],
            'playeddevcard': [46, 47],
            'players': [47, 211],
        }

        actions_metadata: SocDataMetadata = {
            'actions': [0, soc_data.ACTION_SIZE],
        }

        return (spatial_metadata, linear_metadata, actions_metadata)
