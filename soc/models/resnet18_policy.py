import argparse
from torch import nn
from .resnet18 import conv1x1, Bottleneck, BasicBlock
from .hexa_conv import HexaConv2d


class ResNet18Policy(nn.Module):
    def __init__(
        self,
        config,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None
    ):
        super(ResNet18Policy, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        data_input_size = config['data_input_size']
        self.inplanes = data_input_size[0]

        self.spatial_state_output_size = config['data_output_size'][0]
        self.state_output_size = config['data_output_size'][1]
        self.action_output_size = config['data_output_size'][2]
        self.n_core_planes = 32
        self.n_core_outputs = self.n_core_planes * self.spatial_state_output_size[
            1] * self.spatial_state_output_size[2]
        self.n_spatial_planes = self.spatial_state_output_size[0]
        self.n_states = self.state_output_size[0] * self.state_output_size[1]
        self.n_actions = self.action_output_size[0] * self.action_output_size[1]

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = HexaConv2d(
            self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 256, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1, dilate=False)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=1, dilate=False)
        self.layer4 = self._make_layer(block, self.n_core_planes, layers[3], stride=1, dilate=False)
        self.spatial_state_head = HexaConv2d(
            self.n_core_planes,
            self.n_spatial_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.linear_state_head = nn.Linear(self.n_core_outputs, self.n_states)
        self.policy_head = nn.Linear(self.n_core_outputs, self.n_actions)

        for m in self.modules():
            if isinstance(m, HexaConv2d):
                continue
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an
        # identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )

        return nn.Sequential(*layers)

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        return parser

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        y_spatial_state = self.spatial_state_head(x)  # bs x n_core_planes x H x W

        bs = x.shape[0]
        y_linear = x.view(bs, -1)
        y_state = self.linear_state_head(y_linear)
        y_action_logits = self.policy_head(y_linear)
        y_action_logits_reshaped = y_action_logits.reshape(bs * self.action_output_size[0], -1)

        return y_spatial_state, y_state, y_action_logits_reshaped

    def forward(self, x):
        return self._forward_impl(x)
