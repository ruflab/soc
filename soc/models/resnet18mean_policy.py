import torch
from torch import nn
from omegaconf import OmegaConf, DictConfig
from .resnet18 import conv1x1, Bottleneck, BasicBlock
from .hexa_conv import HexaConv2d


class ResNet18MeanConcatPolicy(nn.Module):
    def __init__(
        self,
        omegaConf: DictConfig,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None
    ):
        super(ResNet18MeanConcatPolicy, self).__init__()
        if norm_layer is None:
            # Batch norm does not fit well with regression.
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.InstanceNorm2d
            # norm_layer = nn.GroupNorm
        self._norm_layer = norm_layer

        # When we are here, the config has already been checked by OmegaConf
        # so we can extract primitives to use with other libs
        conf = OmegaConf.to_container(omegaConf)
        assert isinstance(conf, dict)

        data_input_size = conf['data_input_size']
        self.game_input_size = data_input_size[0]  # [S_h, C, H, W]
        assert len(self.game_input_size) == 4

        self.text_input_size = data_input_size[1]  # [S_h, S_text, F_bert]
        assert len(self.text_input_size) == 3

        self.inplanes = self.game_input_size[0] * self.game_input_size[1]

        self.n_core_planes = conf['n_core_planes']
        self.n_core_outputs = self.n_core_planes * self.game_input_size[2] * self.game_input_size[3]

        data_output_size = conf['data_output_size']
        self.spatial_state_output_size = data_output_size[0]
        self.state_output_size = data_output_size[1]
        self.action_output_size = data_output_size[2]
        n_future_seq = self.spatial_state_output_size[0]

        self.n_spatial_planes = n_future_seq * self.spatial_state_output_size[1]
        self.n_states = n_future_seq * self.state_output_size[1]
        self.n_actions = n_future_seq * self.action_output_size[1]

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
            self.inplanes, 32 * self.n_core_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.inplanes = 32 * self.n_core_planes

        if norm_layer == nn.GroupNorm:
            self.bn1 = norm_layer(4, self.inplanes)
        else:
            self.bn1 = norm_layer(self.inplanes)
        self.act_f = nn.GELU()

        self.layer1 = self._make_layer(
            block, 8 * self.n_core_planes, layers[0], stride=1, dilate=False
        )
        self.layer2 = self._make_layer(
            block, 4 * self.n_core_planes, layers[1], stride=1, dilate=False
        )
        self.layer3 = self._make_layer(
            block, 2 * self.n_core_planes, layers[2], stride=1, dilate=False
        )
        self.layer4 = self._make_layer(
            block, 1 * self.n_core_planes, layers[3], stride=1, dilate=False
        )

        # Game feature extractor
        self.cnn = nn.Sequential(
            self.conv1,
            self.bn1,
            self.act_f,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )

        # Text feature extractor
        # We use a simple mean operation
        self.extractor = nn.Sequential()

        # Fusion module
        # We use a simple concatenation operation
        self.fusion = nn.Sequential()

        # Heads
        F_text = self.text_input_size[-1]
        self.spatial_state_head = nn.Sequential(
            nn.Conv2d(
                self.n_core_planes,
                self.n_spatial_planes * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.GELU(),
            nn.Conv2d(
                self.n_spatial_planes * 2,
                self.n_spatial_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
        )

        self.linear_state_head = nn.Sequential(
            nn.Linear(self.n_core_outputs + F_text, 512), nn.GELU(), nn.Linear(512, self.n_states)
        )

        self.policy_head = nn.Sequential(
            nn.Linear(self.n_core_outputs + F_text, 512),
            nn.GELU(),
            nn.Linear(512, self.n_actions),
        )

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
            if norm_layer == nn.GroupNorm:
                n1 = norm_layer(4, planes * block.expansion)
            else:
                n1 = norm_layer(planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                n1,
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

    def _forward_impl(self, x, x_text, x_text_mask):
        bs, S, C, H, W = x.shape

        # Game feature extraction
        x = x.view(bs, S * C, H, W)
        z = self.cnn(x)
        z_linear = z.reshape(bs, -1)

        # Text feature extraction
        x_text = x_text[:, -1]  # [bs, S_text, F_bert]
        x_text_mask = x_text_mask[:, -1]  # [bs, S_text]
        x_text = torch.sum(x_text * x_text_mask.unsqueeze(-1), dim=1)  # [bs, F_bert]
        x_text_f = x_text / (torch.sum(x_text_mask, dim=1, keepdim=True) + 1e-7)

        # Fusion
        z_linear = torch.cat([z_linear, x_text_f], dim=1)
        z_linear = self.fusion(z_linear)

        # Heads
        y_spatial_state_logits = self.spatial_state_head(z)
        y_spatial_state_logits_seq = y_spatial_state_logits.reshape([
            bs,
        ] + self.spatial_state_output_size)

        y_state_logits = self.linear_state_head(z_linear)
        y_state_logits_seq = y_state_logits.reshape([
            bs,
        ] + self.state_output_size)

        y_action_logits = self.policy_head(z_linear)
        y_action_logits_seq = y_action_logits.reshape([
            bs,
        ] + self.action_output_size)

        return y_spatial_state_logits_seq, y_state_logits_seq, y_action_logits_seq

    def forward(self, x, x_text, x_text_mask):
        return self._forward_impl(x, x_text, x_text_mask)

    def _forward_bypass_text_impl(self, x):
        bs, S, C, H, W = x.shape

        # Game feature extraction
        x = x.view(bs, S * C, H, W)
        z = self.cnn(x)
        z_linear = z.reshape(bs, -1)

        # Fake text feature extraction
        x_text_f = torch.zeros(bs, self.text_input_size[1])

        # Fusion
        z_linear = torch.cat([z_linear, x_text_f], dim=1)
        z_linear = self.fusion(z_linear)

        # Heads
        y_spatial_state_logits = self.spatial_state_head(z_linear.reshape(z.shape))
        y_spatial_state_logits_seq = y_spatial_state_logits.reshape([
            bs,
        ] + self.spatial_state_output_size)

        y_state_logits = self.linear_state_head(z_linear)
        y_state_logits_seq = y_state_logits.reshape([
            bs,
        ] + self.state_output_size)

        y_action_logits = self.policy_head(z_linear)
        y_action_logits_seq = y_action_logits.reshape([
            bs,
        ] + self.action_output_size)

        return y_spatial_state_logits_seq, y_state_logits_seq, y_action_logits_seq

    def forward_bypass_text(self, x, x_text, x_text_mask):
        return self._forward_bypass_text_impl(x)


class ResNet18MeanFFPolicy(ResNet18MeanConcatPolicy):
    def __init__(
        self,
        omegaConf: DictConfig,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None
    ):
        super(ResNet18MeanConcatPolicy, self).__init__()
        if norm_layer is None:
            # Batch norm does not fit well with regression.
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.InstanceNorm2d
            # norm_layer = nn.GroupNorm
        self._norm_layer = norm_layer

        # When we are here, the config has already been checked by OmegaConf
        # so we can extract primitives to use with other libs
        conf = OmegaConf.to_container(omegaConf)
        assert isinstance(conf, dict)

        data_input_size = conf['data_input_size']
        self.game_input_size = data_input_size[0]  # [S_h, C, H, W]
        assert len(self.game_input_size) == 4

        self.text_input_size = data_input_size[1]  # [S_h, S_text, F_bert]
        assert len(self.text_input_size) == 3

        self.inplanes = self.game_input_size[0] * self.game_input_size[1]

        self.n_core_planes = conf['n_core_planes']
        self.n_core_outputs = self.n_core_planes * self.game_input_size[2] * self.game_input_size[3]

        data_output_size = conf['data_output_size']
        self.spatial_state_output_size = data_output_size[0]
        self.state_output_size = data_output_size[1]
        self.action_output_size = data_output_size[2]
        n_future_seq = self.spatial_state_output_size[0]

        self.n_spatial_planes = n_future_seq * self.spatial_state_output_size[1]
        self.n_states = n_future_seq * self.state_output_size[1]
        self.n_actions = n_future_seq * self.action_output_size[1]

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
            self.inplanes, 32 * self.n_core_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.inplanes = 32 * self.n_core_planes

        if norm_layer == nn.GroupNorm:
            self.bn1 = norm_layer(4, self.inplanes)
        else:
            self.bn1 = norm_layer(self.inplanes)
        self.act_f = nn.GELU()

        self.layer1 = self._make_layer(
            block, 8 * self.n_core_planes, layers[0], stride=1, dilate=False
        )
        self.layer2 = self._make_layer(
            block, 4 * self.n_core_planes, layers[1], stride=1, dilate=False
        )
        self.layer3 = self._make_layer(
            block, 2 * self.n_core_planes, layers[2], stride=1, dilate=False
        )
        self.layer4 = self._make_layer(
            block, 1 * self.n_core_planes, layers[3], stride=1, dilate=False
        )

        # Game feature extractor
        self.cnn = nn.Sequential(
            self.conv1,
            self.bn1,
            self.act_f,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )

        # Text feature extractor
        # We use a simple mean operation
        self.extractor = nn.Sequential()

        # Fusion module
        self.fusion = nn.Sequential(
            nn.Linear(self.n_core_outputs + self.text_input_size[-1], 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, self.n_core_outputs)
        )

        # Heads
        self.spatial_state_head = nn.Sequential(
            nn.Conv2d(
                self.n_core_planes,
                self.n_spatial_planes * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.GELU(),
            nn.Conv2d(
                self.n_spatial_planes * 2,
                self.n_spatial_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
        )

        self.linear_state_head = nn.Sequential(
            nn.Linear(self.n_core_outputs, 512), nn.GELU(), nn.Linear(512, self.n_states)
        )

        self.policy_head = nn.Sequential(
            nn.Linear(self.n_core_outputs, 512),
            nn.GELU(),
            nn.Linear(512, self.n_actions),
        )

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


class ResNet18MeanFFResPolicy(ResNet18MeanFFPolicy):
    def _forward_impl(self, x, x_text, x_text_mask):
        bs, S, C, H, W = x.shape

        # Game feature extraction
        x = x.view(bs, S * C, H, W)
        z = self.cnn(x)
        z_linear = z.reshape(bs, -1)

        # Text feature extraction
        x_text = x_text[:, -1]  # [bs, S_text, F_bert]
        x_text_mask = x_text_mask[:, -1]  # [bs, S_text]
        x_text = torch.sum(x_text * x_text_mask.unsqueeze(-1), dim=1)  # [bs, F_bert]
        x_text_f = x_text / (torch.sum(x_text_mask, dim=1, keepdim=True) + 1e-7)

        # Residual fusion
        z_linear_fus = torch.cat([z_linear, x_text_f], dim=1)
        batch_mask = (torch.sum(x_text_mask, dim=1, keepdims=True) != 0) * 1.
        z_linear = z_linear + self.fusion(z_linear_fus) * batch_mask

        # Heads
        y_spatial_state_logits = self.spatial_state_head(z_linear.reshape(z.shape))
        y_spatial_state_logits_seq = y_spatial_state_logits.reshape([
            bs,
        ] + self.spatial_state_output_size)

        y_state_logits = self.linear_state_head(z_linear)
        y_state_logits_seq = y_state_logits.reshape([
            bs,
        ] + self.state_output_size)

        y_action_logits = self.policy_head(z_linear)
        y_action_logits_seq = y_action_logits.reshape([
            bs,
        ] + self.action_output_size)

        return y_spatial_state_logits_seq, y_state_logits_seq, y_action_logits_seq
