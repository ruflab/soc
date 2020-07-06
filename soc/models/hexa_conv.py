import torch
from torch import nn


class HexaConv2d(nn.Conv2d):
    def __init__(self, *args, **kargs):
        super(HexaConv2d, self).__init__(*args, **kargs)

        self.mask = nn.Parameter(self.get_mask(), requires_grad=False)

        copy_w = self.weight.clone().detach()
        self.weight = nn.Parameter(copy_w * self.mask)

        this = self
        self.weight.register_hook(lambda grad: grad * this.mask)

    def get_mask(self):
        _, _, kh, kw = self.weight.shape
        if kh < 3 or kw < 3:
            raise Exception("A hexaconv must have a size a height and a width >= 3x3")

        if kh % 2 != 1 or kw % 2 != 1:
            raise Exception("A hexaconv must have a kernel where its width and height are odds")

        if kh != kw:
            raise Exception("A hexaconv must have a square kernel")

        N = kh

        mask = torch.ones_like(self.weight, requires_grad=False)
        for i in range(N):
            idx = ((N + 1) // 2 + i) % N

            if i < (N - 1) / 2:
                mask[:, :, i, idx:] = 0

            if i > (N - 1) / 2:
                mask[:, :, i, :idx] = 0

        return mask


class HexaConv3d(nn.Conv3d):
    def __init__(self, *args, **kargs):
        super(HexaConv3d, self).__init__(*args, **kargs)

        self.mask = nn.Parameter(self.get_mask(), requires_grad=False)

        copy_w = self.weight.clone().detach()
        self.weight = nn.Parameter(copy_w * self.mask)

        this = self
        self.weight.register_hook(lambda grad: grad * this.mask)

    def get_mask(self):
        _, _, kd, kh, kw = self.weight.shape
        if kh < 3 or kw < 3:
            raise Exception("A hexaconv must have a size a height and a width >= 3x3")

        if kh % 2 != 1 or kw % 2 != 1:
            raise Exception("A hexaconv must have a kernel where its width and height are odds")

        if kh != kw:
            raise Exception("A hexaconv must have a square kernel")

        N = kh

        mask = torch.ones_like(self.weight, requires_grad=False)
        for i in range(N):
            idx = ((N + 1) // 2 + i) % N

            if i < (N - 1) / 2:
                mask[:, :, :, i, idx:] = 0

            if i > (N - 1) / 2:
                mask[:, :, :, i, :idx] = 0

        return mask
