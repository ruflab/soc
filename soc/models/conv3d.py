import argparse
import math
import torch.nn as nn
from .. import utils
from .hexa_conv import HexaConv3d
from typing import List, Tuple


class Conv3dModel(nn.Module):
    """
        3d convolution modèle

        Notes:
        To use as a predictive coding, you must ensure to pad the input correctly:
            ex: for a 3xHxW kernel, pad by 2 before, 0 after the seq.
                This is achieved with padding (left, right, top, bottom, 2, 0)

    """
    def __init__(self, config):
        super(Conv3dModel, self).__init__()

        self.data_input_size = config['data_input_size']
        self.data_output_size = config['data_output_size']
        self.num_layers = config.get('num_layers', 2)
        self.h_chan_dim = self._extend_for_multilayer(config.get('h_chan_dim', 32), self.num_layers)
        self.kernel_size = self._extend_for_multilayer(
            config.get('kernel_size', (3, 3, 3)), self.num_layers
        )
        self.strides = self._extend_for_multilayer(
            config.get('strides', (1, 1, 1)), self.num_layers
        )
        self.paddings = self._extend_for_multilayer(
            config.get('paddings', (1, 1, 1, 1, 2, 0)), self.num_layers
        )
        assert len(self.kernel_size) == self.num_layers

        layers = []
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

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            '--h_chan_dim',
            type=int,
            nargs='+',
            default=argparse.SUPPRESS,
            help='List of hidden channels per layer'
        )
        parser.add_argument(
            '--kernel_size',
            type=utils.soc_tuple,
            nargs='+',
            default=argparse.SUPPRESS,
            help='List of Kernel size per layer'
        )
        parser.add_argument(
            '--num_layers', type=int, default=argparse.SUPPRESS, help='Number of layers'
        )
        parser.add_argument(
            '--strides',
            type=utils.soc_tuple,
            nargs='+',
            default=argparse.SUPPRESS,
            help='List of Kernel size per layer'
        )
        parser.add_argument(
            '--paddings',
            type=utils.soc_tuple,
            nargs='+',
            default=argparse.SUPPRESS,
            help='List of Kernel size per layer'
        )

        return parser

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

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
