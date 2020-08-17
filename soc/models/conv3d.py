import math
import torch.nn as nn
from dataclasses import dataclass, field
from omegaconf import MISSING, OmegaConf, DictConfig
from typing import List, Tuple, Any
from .hexa_conv import HexaConv3d


@dataclass
class Conv3dModelConfig:
    data_input_size: List[int] = MISSING
    data_output_size: List[int] = MISSING

    name: str = 'Conv3dModel'
    num_layers: int = 2
    h_chan_dim: List[int] = field(default_factory=lambda: [32] * 2)
    kernel_size: List[Any] = field(default_factory=lambda: [(3, 3, 3)] * 2)
    strides: List[Any] = field(default_factory=lambda: [(1, 1, 1)] * 2)
    paddings: List[Any] = field(default_factory=lambda: [(1, 1, 1, 1, 2, 0)] * 2)


class Conv3dModel(nn.Module):
    """
        3d convolution modèle

        Notes:
        To use as a predictive coding, you must ensure to pad the input correctly:
            ex: for a 3xHxW kernel, pad by 2 before, 0 after the seq.
                This is achieved with padding (left, right, top, bottom, 2, 0)

    """
    def __init__(self, omegaConf: DictConfig):
        super(Conv3dModel, self).__init__()

        # When we are here, the config has already been checked by OmegaConf
        # so we can extract primitives to use with other libs
        conf = OmegaConf.to_container(omegaConf)
        assert isinstance(conf, dict)

        self.data_input_size = conf['data_input_size']
        self.data_output_size = conf['data_output_size']
        self.num_layers = conf['num_layers']
        self.kernel_size = self._extend_for_multilayer(conf['kernel_size'])
        self.check_kernel_size()
        self.h_chan_dim = self._extend_for_multilayer(conf['h_chan_dim'])
        self.check_h_chan_dim()
        self.strides = self._extend_for_multilayer(conf['strides'])
        self.check_strides()
        self.paddings = self._extend_for_multilayer(conf['paddings'])
        self.check_paddings()

        layers: List[nn.Module] = []
        for i in range(self.num_layers - 1):
            layers.append(nn.ConstantPad3d(self.paddings[i], 0))
            layers.append(
                HexaConv3d(
                    self.data_input_size[0] if i == 0 else self.h_chan_dim[i - 1],
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
                self.h_chan_dim[-1],
                self.data_output_size[0],
                self.kernel_size[-1],
                stride=self.strides[-1],
                padding=0
            )
        )

        self.m = nn.Sequential(*layers)

    def forward(self, input_tensor):
        """
            Input for 3D convolution should be like this:
                (Bs, C_in, Depth, H, W)

            We assume the input is formated like this:
                (Bs, S, C_in, H, W)

            The Sequence dim is moved to align with the Depth dim of the 3d conv
        """

        # (Bs, S, C_in, H, W) -> (Bs, C_in, Depth, H, W)
        tensor = input_tensor.permute(0, 2, 1, 3, 4)

        # Note: the output tensor dimensions depends on the strides
        # of the model. To get the same number of steps as the input_tensor
        # You need to keep a stride of 1 for all layers with the right padding
        out = self.m(tensor)

        output_tensor = out.permute(0, 2, 1, 3, 4)

        return (output_tensor, )

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

        C_out = self.data_output_size[0]
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
