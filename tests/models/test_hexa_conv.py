import os
import unittest
import numpy as np
import torch
from soc.models import HexaConv2d

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestHexaConv2d(unittest.TestCase):
    def test_get_mask_3_3(self):
        c_in = 1
        c_out = 1
        kernel_size = 3

        conv = HexaConv2d(c_in, c_out, kernel_size)

        mask = conv.get_mask()

        y = torch.ones(c_out, c_in, kernel_size, kernel_size)
        y[:, :, 0, 2] = 0
        y[:, :, 2, 0] = 0

        np.testing.assert_array_equal(mask, y)

    def test_get_mask_5_5(self):
        c_in = 1
        c_out = 1
        kernel_size = 5

        conv = HexaConv2d(c_in, c_out, kernel_size)

        mask = conv.get_mask()

        y = torch.ones(c_out, c_in, kernel_size, kernel_size)
        y[:, :, 0, 3] = 0
        y[:, :, 0, 4] = 0
        y[:, :, 1, 4] = 0
        y[:, :, 3, 0] = 0
        y[:, :, 4, 0] = 0
        y[:, :, 4, 1] = 0

        np.testing.assert_array_equal(mask, y)

    def test_gradients(self):
        bs = 2
        c_in = 1
        c_out = 1
        kernel_size = 3

        model = HexaConv2d(c_in, c_out, kernel_size)
        optim = torch.optim.SGD(model.parameters(), 1.)

        in_data = torch.ones(bs, c_in, 3, 3)
        scalar = model(in_data)
        loss = torch.mean(scalar**2)

        model.zero_grad()
        loss.backward()
        optim.step()

        assert model.weight[:, :, 0, 2].sum() == 0
        assert model.weight[:, :, 2, 0].sum() == 0
        assert model.weight.grad[:, :, 0, 2].sum() == 0
        assert model.weight.grad[:, :, 2, 0].sum() == 0
