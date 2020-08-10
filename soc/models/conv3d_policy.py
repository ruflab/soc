import math
import torch.nn as nn
from omegaconf import OmegaConf
from typing import List, Tuple
from .hexa_conv import HexaConv3d
from .conv3d import Conv3dModelConfig


class Conv3dModelPolicy(nn.Module):
    """
        3d convolution modèle

        Notes:
        To use as a predictive coding, you must ensure to pad the input correctly:
            ex: for a 3xHxW kernel, pad by 2 before, 0 after the seq.
                This is achieved with padding (left, right, top, bottom, 2, 0)

    """
    def __init__(self, config: Conv3dModelConfig):
        super(Conv3dModelPolicy, self).__init__()

        # When we are here, the config has already been checked by OmegaConf
        # so we can extract primitives to use with other libs
        conf = OmegaConf.to_container(config)
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
        self.check_kernel_size()
        self.h_chan_dim = self._extend_for_multilayer(conf['h_chan_dim'])
        self.check_h_chan_dim()
        self.strides = self._extend_for_multilayer(conf['strides'])
        self.check_strides()
        self.paddings = self._extend_for_multilayer(conf['paddings'])
        self.check_paddings()

        self.n_core_planes = self.h_chan_dim[-1]
        self.n_core_outputs = self.n_core_planes * self.data_input_size[2] * self.data_input_size[3]
        self.head_hidden_size = 512

        layers: List[nn.Module] = []
        for i in range(self.num_layers - 1):
            layers.append(nn.ConstantPad3d(self.paddings[i], 0))
            layers.append(
                HexaConv3d(
                    self.data_input_size[1] if i == 0 else self.h_chan_dim[i - 1],
                    self.h_chan_dim[i],
                    self.kernel_size[i],
                    stride=self.strides[i],
                    padding=0
                )
            )
            layers.append(nn.ReLU(True))

        layers.append(nn.ConstantPad3d(self.paddings[-1], 0))
        layers.append(
            HexaConv3d(
                self.h_chan_dim[-2],
                self.h_chan_dim[-1],
                self.kernel_size[-1],
                stride=self.strides[-1],
                padding=0
            )
        )

        self.m = nn.Sequential(*layers)

        self.spatial_state_head = nn.Sequential(
            nn.Conv3d(
                self.n_core_planes,
                self.n_spatial_planes * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.Conv3d(
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

    def forward(self, input_tensor):
        """
            Input for 3D convolution should be like this:
                (Bs, C_in, Depth, H, W)

            We assume the input is formated like this:
                (Bs, S, C_in, H, W)

            The Sequence dim is moved to align with the Depth dim of the 3d conv
        """

        # (Bs, S, C_in, H, W) -> (Bs, C_in, Depth, H, W)
        bs, S, C, H, W = input_tensor.shape
        tensor = input_tensor.permute(0, 2, 1, 3, 4)

        # Note: the output tensor dimensions depends on the strides
        # of the model. To get the same number of steps as the input_tensor
        # You need to keep a stride of 1 for all layers with the right padding
        out = self.m(tensor)
        y = out.permute(0, 2, 1, 3, 4)

        y_linear = y.reshape(bs * S, -1)

        y_spatial_state_logits = self.spatial_state_head(out)
        y_spatial_state_logits.permute(0, 2, 1, 3, 4)
        y_spatial_state_logits_seq = y_spatial_state_logits.reshape([bs, S] + self
                                                                    .spatial_state_output_size[1:])
        y_state_logits = self.linear_state_head(y_linear)
        y_state_logits_seq = y_state_logits.reshape([bs, S] + self.state_output_size[1:])

        y_action_logits = self.policy_head(y_linear)
        y_action_logits_seq = y_action_logits.reshape([bs, S] + self.action_output_size[1:])

        outputs = (y_spatial_state_logits_seq, y_state_logits_seq, y_action_logits_seq)

        return (outputs, )

    def get_output_dim(self, input_dim: List) -> Tuple:
        """Return the output shape for a given input shape."""
        if len(input_dim) == 5:
            # Contains the batch_size
            D_in = input_dim[1]
            H_in = input_dim[3]
            W_in = input_dim[4]
        else:
            D_in = input_dim[0]
            H_in = input_dim[2]
            W_in = input_dim[3]

        C_out = self.n_core_planes
        D_out = D_in
        H_out = H_in
        W_out = W_in
        for i in range(self.num_layers):
            ith_padding = self.paddings[i]
            if type(ith_padding) is int:
                s_pad = h_pad = w_pad = ith_padding
            else:
                w_pad = ith_padding[0] + ith_padding[1]
                h_pad = ith_padding[2] + ith_padding[3]
                s_pad = ith_padding[4] + ith_padding[5]

            D_out = math.floor((D_out + s_pad - (self.kernel_size[i][0] - 1) - 1)
                               / self.strides[i][0] + 1)
            H_out = math.floor((H_out + h_pad - (self.kernel_size[i][1] - 1) - 1)
                               / self.strides[i][1] + 1)
            W_out = math.floor((W_out + w_pad - (self.kernel_size[i][2] - 1) - 1)
                               / self.strides[i][2] + 1)
        if len(input_dim) == 5:
            return (input_dim[0], D_out, C_out, H_out, W_out)
        else:
            return (D_out, C_out, H_out, W_out)

    def check_kernel_size(self):
        if not isinstance(self.kernel_size, list):
            raise ValueError('`self.kernel_size` must be a list of list of 3 ints')
        if not all([isinstance(x, list) and len(x) == 3 for x in self.kernel_size]):
            raise ValueError('`self.kernel_size` must be a list of list of 3 ints')

    def check_strides(self):
        if not isinstance(self.strides, list):
            raise ValueError('`self.strides` must be a list of list of 3 ints')
        if not all([isinstance(x, list) and len(x) == 3 for x in self.strides]):
            raise ValueError('`self.strides` must be a list of list of 3 ints')

    def check_paddings(self):
        if not isinstance(self.paddings, list):
            raise ValueError('`self.paddings` must be a list of list of 6 ints')
        if not all([isinstance(x, list) and len(x) == 6 for x in self.paddings]):
            raise ValueError('`self.paddings` must be a list of list of 6 ints')

    def check_h_chan_dim(self):
        if not isinstance(self.h_chan_dim, list):
            raise ValueError('`self.h_chan_dim` must be a list of list of 2 ints')

        if not all([type(x) == int for x in self.h_chan_dim]):
            raise ValueError('`self.h_chan_dim` must be a list of ints')

    def _extend_for_multilayer(self, param: list):
        if len(param) == 1:
            param = [param[0]] * self.num_layers

        if len(param) != self.num_layers:
            raise ValueError('`param` list should be of size {}'.format(self.num_layers))

        return param
