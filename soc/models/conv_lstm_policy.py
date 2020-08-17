###
# Taken from https://raw.githubusercontent.com/ndrplz/ConvLSTM_pytorch/master/convlstm.py
###
import torch.nn as nn
import torch
from omegaconf import OmegaConf, DictConfig
from .conv_lstm import ConvLSTMCell


class ConvLSTMPolicy(nn.Module):
    """

    Parameters:
        config: Dict containing
            input_dim: Number of channels in input
            h_chan_dim: Number of hidden channels
            kernel_size: Size of kernel in convolutions
            num_layers: Number of LSTM layers stacked on each other
            batch_first: Whether or not dimension 0 is the batch or not
            bias: Bias or no bias in Convolution
            return_all_layers: Return the list of computations for all layers

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTMPolicy(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """
    def __init__(self, omegaConf: DictConfig):
        super(ConvLSTMPolicy, self).__init__()

        # When we are here, the config has already been checked by OmegaConf
        # so we can extract primitives to use with other libs
        conf = OmegaConf.to_container(omegaConf)
        assert isinstance(conf, dict)

        self.data_input_size = conf['data_input_size']

        data_output_size = conf['data_output_size']
        self.spatial_state_output_size = data_output_size[0]
        self.state_output_size = data_output_size[1]
        self.action_output_size = data_output_size[2]

        self.n_spatial_planes = self.spatial_state_output_size[1]
        self.n_states = self.state_output_size[1]
        self.n_actions = self.action_output_size[1]

        self.num_layers = conf['num_layers']
        self.kernel_size = self._extend_for_multilayer(conf['kernel_size'])
        self.h_chan_dim = self._extend_for_multilayer(conf['h_chan_dim'])
        self.batch_first = conf['batch_first']
        self.bias = conf['bias']
        self.return_all_layers = conf['return_all_layers']

        self.n_core_planes = self.h_chan_dim[-1]
        self.n_core_outputs = self.n_core_planes * self.data_input_size[2] * self.data_input_size[3]
        self.head_hidden_size = 512

        cell_list = []
        for i in range(0, self.num_layers):
            cur_in_chan_dim = self.data_input_size[1] if i == 0 else self.h_chan_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_in_chan_dim,
                    h_chan_dim=self.h_chan_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

        self.spatial_state_head = nn.Sequential(
            nn.Conv2d(
                self.n_core_planes,
                self.n_spatial_planes * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.n_spatial_planes * 2,
                self.n_spatial_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
        )
        self.linear_state_head = nn.Sequential(
            nn.Linear(self.n_core_outputs, self.head_hidden_size),
            nn.ReLU(),
            nn.Linear(self.head_hidden_size, self.n_states)
        )

        self.policy_head = nn.Sequential(
            nn.Linear(self.n_core_outputs, self.head_hidden_size),
            nn.ReLU(),
            nn.Linear(self.head_hidden_size, self.n_actions),
        )

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        bs, S, C, H, W = input_tensor.shape

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=bs, image_size=(H, W))

        layer_output_list = []
        last_state_list = []
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(S):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        y = layer_output_list[-1]
        y_linear = y.reshape(bs * S, -1)

        y_spatial_shape = [bs * S, self.n_core_planes] + self.spatial_state_output_size[-2:]
        y_spatial_state_logits = self.spatial_state_head(y.reshape(y_spatial_shape))
        y_spatial_state_logits_seq = y_spatial_state_logits.reshape([bs, S] + self
                                                                    .spatial_state_output_size[1:])
        y_state_logits = self.linear_state_head(y_linear)
        y_state_logits_seq = y_state_logits.reshape([bs, S] + self.state_output_size[1:])

        y_action_logits = self.policy_head(y_linear)
        y_action_logits_seq = y_action_logits.reshape([bs, S] + self.action_output_size[1:])

        outputs = (y_spatial_state_logits_seq, y_state_logits_seq, y_action_logits_seq)

        return outputs, layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    def _extend_for_multilayer(self, param: list):
        if len(param) == 1:
            param = [param[0]] * self.num_layers

        if len(param) != self.num_layers:
            raise ValueError('`param` list should be of size {}'.format(self.num_layers))

        return param
