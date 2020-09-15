from dataclasses import dataclass
import torch
from torch import nn
from omegaconf import OmegaConf, DictConfig
from .resnet18 import conv1x1, Bottleneck, BasicBlock
from .resnet18 import ResNetConfig
from .hexa_conv import HexaConv2d
from .hopfield import Hopfield


@dataclass
class ResNetFusionConfig(ResNetConfig):
    fusion_num_heads: int = 8
    beta: float = 0.3
    update_steps_max: int = 0
    self_att_fusion: bool = True


class ResNet18FusionPolicy(nn.Module):
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
        super(ResNet18FusionPolicy, self).__init__()
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
        self.game_input_size = data_input_size[0]
        self.text_input_size = data_input_size[1]
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
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 8 * self.n_core_planes, layers[0])
        self.layer2 = self._make_layer(
            block, 4 * self.n_core_planes, layers[1], stride=1, dilate=False
        )
        self.layer3 = self._make_layer(
            block, 2 * self.n_core_planes, layers[2], stride=1, dilate=False
        )
        self.layer4 = self._make_layer(block, self.n_core_planes, layers[3], stride=1, dilate=False)

        self.cnn = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )

        # Text feature extractor
        # We use a simple mean operation
        self.extractor = nn.Sequential()

        # Fusion module
        # Our state that we want to change is the CNN state
        self.self_att_fusion = conf['self_att_fusion']
        if self.self_att_fusion is True:
            self.text_projection = nn.Linear(self.text_input_size[-1], self.n_core_planes)
            self.fusion = Hopfield(
                input_size=self.n_core_planes,
                hidden_size=1024,
                num_heads=conf['fusion_num_heads'],
                scaling=conf['beta'
                             ],  # Beta parameter, can be set as a tensor to assign different betas
                update_steps_max=conf['update_steps_max'],
                dropout=0.,
                batch_first=True,
                association_activation=None,
            )
        else:
            self.fusion = Hopfield(
                input_size=self.n_core_outputs,
                stored_pattern_size=self.text_input_size[-1],
                pattern_projection_size=self.text_input_size[-1],
                output_size=self.n_core_outputs,
                pattern_projection_as_connected=True,
                hidden_size=self.n_core_outputs,
                num_heads=conf['fusion_num_heads'],
                scaling=conf['beta'
                             ],  # Beta parameter, can be set as a tensor to assign different betas
                update_steps_max=conf['update_steps_max'],
                dropout=0.,
                batch_first=True,
                association_activation=None,
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
            nn.ReLU(),
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
            nn.Linear(self.n_core_outputs, 512), nn.ReLU(), nn.Linear(512, self.n_states)
        )

        self.policy_head = nn.Sequential(
            nn.Linear(self.n_core_outputs, 512),
            nn.ReLU(),
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
        bs, S_hist, C, H, W = x.shape
        # bs, S_hist, S_text, F_bert = x_text.shape

        x = x.view(bs, S_hist * C, H, W)
        # See note [TorchScript super()]
        z = self.cnn(x)
        _, C_core, H_core, W_core = z.shape  # bs, C_core, H_core, W_core

        # Fusion
        # Input to hopfield network (K, Q, V)
        if self.self_att_fusion is True:
            n_z = H_core * W_core
            d_z = self.n_core_planes

            # We associate (contextualized) game data with text data
            # Extraction and fusion happen at the same time
            x_text = x_text[:, -1]  # [bs, S_Text, F_bert]
            x_text = self.text_projection(x_text)  # [bs, S_Text, F_text]
            x_text_mask = x_text_mask[:, -1]  # [bs, S_Text]
            x_text_mask = torch.cat([torch.ones((bs, n_z)), x_text_mask],
                                    dim=1).to(torch.bool)  # [bs, n_z + S_text]
            x_text_mask = ~x_text_mask  # Negation (True value will be filled with -inf)

            input_data = torch.cat([z.view(bs, n_z, d_z), x_text.view(bs, -1, d_z)], dim=1)
            z = self.fusion(
                input_data,  # [bs, n_z + S_text, d_z]
                stored_pattern_padding_mask=x_text_mask
            )
            z = z[:, 0:n_z].reshape((bs, C_core, H_core, W_core))
        else:
            # We use game data to select text data and merge them together
            # Extraction
            x_text = x_text[:, -1]  # [bs, S_text, F_bert]
            x_text_mask = x_text_mask[:, -1].to(torch.bool)  # [bs, S_text]
            x_text_mask = ~x_text_mask  # Negation (True value will be filled with -inf)
            # x_text_mask[:, 0] = True  # Remove CLS Token from the softmax
            input_data = (x_text, z.view(bs, 1, self.n_core_outputs), x_text)
            z_text_modality = self.fusion(input_data, stored_pattern_padding_mask=x_text_mask)
            # Fusion by addition
            z = z + z_text_modality.view((bs, C_core, H_core, W_core))

        # Heads
        z_spatial = z
        z_linear = z.view(bs, -1)
        y_spatial_state_logits = self.spatial_state_head(z_spatial)
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
        bs, S_hist, C, H, W = x.shape
        x = x.view(bs, S_hist * C, H, W)
        # See note [TorchScript super()]
        z = self.cnn(x)
        z_spatial = z
        z_linear = z.reshape(bs, -1)

        y_spatial_state_logits = self.spatial_state_head(z_spatial)
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
