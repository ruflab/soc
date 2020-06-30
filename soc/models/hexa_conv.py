import torch
from torch import nn


class HexaConv2d(nn.Conv2d):
    def __init__(self, *args, **kargs):
        super(HexaConv2d, self).__init__(*args, **kargs)

        mask = self.get_mask()

        def hook_fn(grad):
            return grad * mask

        copy_w = self.weight.clone().detach()
        self.weight = nn.Parameter(copy_w * mask)
        self.weight.register_hook(hook_fn)

    def get_mask(self):
        _, _, kh, kw = self.weight.shape
        if kh < 3 or kw < 3:
            raise Exception("A hexaconv must have a size a height and a width >= 3x3")

        if kh % 2 != 1 or kw % 2 != 1:
            raise Exception("A hexaconv must have a kernel where its width and height are odds")

        if kh != kw:
            raise Exception("A hexaconv must have a square kernel")

        N = kh

        mask = torch.ones_like(self.weight)
        for i in range(N):
            idx = ((N + 1) // 2 + i) % N

            if i < (N - 1) / 2:
                mask[:, :, i, idx:] = 0

            if i > (N - 1) / 2:
                mask[:, :, i, :idx] = 0

        return mask
