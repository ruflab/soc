###
# Taken from https://raw.githubusercontent.com/ndrplz/ConvLSTM_pytorch/master/convlstm.py
###
import torch.nn as nn
import torch
from .hexa_conv import HexaConv2d


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, h_chan_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        h_chan_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.h_chan_dim = h_chan_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = HexaConv2d(
            in_channels=self.input_dim + self.h_chan_dim,
            out_channels=4 * self.h_chan_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.h_chan_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.h_chan_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.h_chan_dim, height, width, device=self.conv.weight.device)
        )


class ConvLSTM(nn.Module):
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
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """
    def __init__(self, config):
        super(ConvLSTM, self).__init__()

        data_input_size = config.get('data_input_size')
        data_output_size = config.get('data_output_size')
        h_chan_dim = config.get('h_chan_dim')
        kernel_size = config.get('kernel_size')
        num_layers = config.get('num_layers')
        batch_first = config.get('batch_first', True)
        bias = config.get('bias', True)
        return_all_layers = config.get('return_all_layers', False)

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `h_chan_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        h_chan_dim = self._extend_for_multilayer(h_chan_dim, num_layers)
        if not len(kernel_size) == len(h_chan_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.data_input_size = data_input_size
        self.data_output_size = data_output_size
        self.h_chan_dim = h_chan_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_in_chan_dim = self.data_input_size[0] if i == 0 else self.h_chan_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_in_chan_dim,
                    h_chan_dim=self.h_chan_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

        self.head = HexaConv2d(
            in_channels=self.h_chan_dim[-1],
            out_channels=self.data_output_size[0],
            kernel_size=self.kernel_size[-1],
            padding=(kernel_size[-1][0] // 2, kernel_size[-1][1] // 2),
            bias=self.bias
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

        b, s, _, height, width = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(height, width))

        layer_output_list = []
        last_state_list = []
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(s):
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

        last_outputs = layer_output_list[-1]
        _, _, chan, _, _ = last_outputs.shape
        outs = self.head(last_outputs.view((b * s, chan, height, width)))
        model_outputs = outs.view((b, s, -1, height, width))

        return model_outputs, layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if isinstance(kernel_size, tuple):
            return

        if isinstance(kernel_size, list):
            if all([isinstance(elem, tuple) for elem in kernel_size]):
                return

        raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    @staticmethod
    def get_default_conf():
        return {
            'h_chan_dim': [128, 128],
            'kernel_size': [(3, 3), (3, 3)],
            'num_layers': 2,
            'bias': True,
            'return_all_layers': False,
        }
