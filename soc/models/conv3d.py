import math
import torch.nn as nn
from typing import List, Tuple


class Conv3dModel(nn.Module):
    def __init__(self, config):
        super(Conv3dModel, self).__init__()

        # breakpoint()
        self.data_input_dims = config.get('data_input_dims')
        self.data_output_dims = config.get('data_output_dims')
        self.num_layers = config.get('num_layers')
        self.h_chan_dim = self._extend_for_multilayer(config.get('h_chan_dim'), self.num_layers)
        self.kernel_size = self._extend_for_multilayer(config.get('kernel_size'), self.num_layers)
        self.strides = self._extend_for_multilayer(config.get('strides'), self.num_layers)
        self.paddings = self._extend_for_multilayer(config.get('paddings'), self.num_layers)
        assert len(self.kernel_size) == self.num_layers

        layers = []
        for i in range(self.num_layers - 1):
            layers.append(
                nn.Conv3d(
                    self.data_input_dims[0] if i == 0 else self.h_chan_dim[i - 1],
                    self.h_chan_dim[i],
                    self.kernel_size[i],
                    stride=self.strides[i],
                    padding=self.paddings[i]
                )
            )
            layers.append(nn.ReLU(True))

        layers.append(
            nn.Conv3d(
                self.h_chan_dim[-1],
                self.data_output_dims[0],
                self.kernel_size[-1],
                stride=self.strides[-1],
                padding=self.paddings[-1]
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

        return output_tensor

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

        C_out = self.data_output_dims[0]
        D_out = D_in
        H_out = H_in
        W_out = W_in
        for i in range(self.num_layers):
            D_out = math.floor((D_out + 2 * self.paddings[i][0] - (self.kernel_size[i][0] - 1) - 1)
                               / self.strides[i][0] + 1)
            H_out = math.floor((H_out + 2 * self.paddings[i][1] - (self.kernel_size[i][1] - 1) - 1)
                               / self.strides[i][1] + 1)
            W_out = math.floor((W_out + 2 * self.paddings[i][2] - (self.kernel_size[i][2] - 1) - 1)
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

    @staticmethod
    def get_default_conf():
        # The first two properties are actually data dependant

        return {
            # 'in_chan_dim': 1,
            # 'output_dim': 1,
            'h_chan_dim': 64,
            'kernel_size': (3, 3, 3),
            'strides': (1, 1, 1),
            'paddings': (1, 1, 1),
            'num_layers': 2,
        }