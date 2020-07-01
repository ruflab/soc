import os
import unittest
import numpy as np
import torch
from soc.models import make_model

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestConv3dModel(unittest.TestCase):
    def test_get_output_dim(self):
        input_dim = (17, 7, 11, 13)  # S, C, H, W
        config = {
            'arch': 'Conv3dModel',
            'data_input_size': (7, 11, 13),
            'data_output_size': (3, 11, 13),
            'num_layers': 2,
            'h_chan_dim': 5,
            'kernel_size': (3, 3, 3),
            'strides': (1, 1, 1),
            'paddings': (1, 1, 2, 2, 3, 3),
        }
        batch_size = 3
        input_tensor = torch.rand((batch_size, ) + input_dim)

        model = make_model(config)
        model.eval()

        out = model(input_tensor)[0]

        x = model.get_output_dim(input_dim)
        y = out.shape[1:]  # We remove the batch size

        np.testing.assert_array_equal(x, y)

    def test_get_output_dim_with_bs(self):
        input_dim = (17, 7, 11, 13)  # S, C, H, W
        config = {
            'arch': 'Conv3dModel',
            'data_input_size': (7, 11, 13),
            'data_output_size': (3, 11, 13),
            'num_layers': 2,
            'h_chan_dim': 5,
            'kernel_size': (3, 3, 3),
            'strides': (1, 1, 1),
            'paddings': (1, 1, 2, 2, 3, 3),
        }
        batch_size = 3
        bs_input_dim = (batch_size, ) + input_dim
        input_tensor = torch.rand(bs_input_dim)

        model = make_model(config)
        model.eval()

        out = model(input_tensor)[0]

        x = model.get_output_dim(bs_input_dim)
        y = out.shape

        np.testing.assert_array_equal(x, y)

    def test_get_output_dim_no_padding(self):
        input_dim = (17, 7, 11, 13)  # S, C, H, W
        config = {
            'arch': 'Conv3dModel',
            'data_input_size': (7, 11, 13),
            'data_output_size': (3, 11, 13),
            'num_layers': 2,
            'h_chan_dim': 5,
            'kernel_size': (3, 3, 3),
            'strides': (1, 1, 1),
            'paddings': 0,
        }
        batch_size = 3
        bs_input_dim = (batch_size, ) + input_dim
        input_tensor = torch.rand(bs_input_dim)

        model = make_model(config)
        model.eval()

        out = model(input_tensor)[0]

        x = model.get_output_dim(input_dim)
        y = out.shape[1:]  # We remove the batch size

        np.testing.assert_array_equal(x, y)
