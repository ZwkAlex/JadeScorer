import math
from typing import OrderedDict

import timm
import torch
from torch import nn
from torchvision import models


def create_pooling_layers():
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        nn.Flatten(1),
    )


class BackBone(nn.Module):
    def __init__(self, model, hook, **kwargs):
        def forward_hook(module, _input, _output):
            if isinstance(_output, (tuple, list)):
                _output = _output[0]
            if torch.is_tensor(_output) and _output.ndim == 4 and self.expected_channels:
                expected = self.expected_channels[len(self.outputs)]
                if _output.shape[1] != expected and _output.shape[-1] == expected:
                    _output = _output.permute(0, 3, 1, 2).contiguous()
            self.outputs.append(_output)

        def get_path(m, hook_str):
            parts = hook_str.split('.')
            path = m
            for part in parts:
                if part.lstrip('-').isdigit():
                    path = path[int(part)]
                elif isinstance(path, OrderedDict):
                    path = path[part]
                else:
                    path = getattr(path, part, None)
            return path

        super().__init__()
        self.backbone = eval(model)
        self.expected_channels = kwargs.get("out_channels", [])
        hook = [hook] if not isinstance(hook, list) else hook
        for h in hook:
            module = get_path(self.backbone, h)
            module.register_forward_hook(forward_hook)
        self.outputs = list()

    def forward(self, x):
        self.outputs.clear()
        _ = self.backbone(x)
        return self.outputs


class PredictHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.GELU(),
            nn.Linear(in_channels // 4, in_channels // 16),
            nn.GELU(),
            nn.Linear(in_channels // 16, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(x) * 10
        # return self.head(x)


class JadeQualityPredictorTwoStream(nn.Module):
    def __init__(self, model_info, model_info2):
        super().__init__()
        self.backbone = BackBone(**model_info)
        self.backbone2 = BackBone(**model_info2)
        out_channels = [192, 384, 768, 1536]
        self.reproject1 = nn.ModuleList(
            nn.Conv2d(c1, c2, kernel_size=(1, 1))
            for c1, c2 in zip(model_info["out_channels"], out_channels))
        self.reproject2 = nn.ModuleList(
            nn.Conv2d(c1, c2, kernel_size=(1, 1))
            for c1, c2 in zip(model_info2["out_channels"], out_channels))

        self.out_conv = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(o * 2, o // 2, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(o // 2, o // 2, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(o // 2, o, kernel_size=(1, 1)),
            ) for o in out_channels
        )

        self.align_linear = nn.ModuleList(
            nn.Linear(o, 1536) for o in out_channels
        )
        self.pool = create_pooling_layers()
        self.color_predict_head = GaussianScoreHead(1536 * 4)
        self.texture_predict_head = GaussianScoreHead(1536 * 4)
        self.freeze_backbones()

    def freeze_backbones(self):
        for backbone in (self.backbone, self.backbone2):
            for p in backbone.parameters():
                p.requires_grad_(False)
            for module in backbone.modules():
                if isinstance(module, nn.LayerNorm):
                    for p in module.parameters():
                        p.requires_grad_(True)

    def unfreeze_last_stages(self, count=1):
        for wrapper in (self.backbone, self.backbone2):
            stages = getattr(wrapper.backbone, "stages", None)
            if stages is None:
                stages = [getattr(wrapper.backbone, n, None) for n in ("layer1", "layer2", "layer3", "layer4")]
                stages = [m for m in stages if m is not None]
            for stage in list(stages)[-max(1, count):]:
                for p in stage.parameters():
                    p.requires_grad_(True)
            # Keep normalization affine parameters trainable during the frozen phase.
            for module in wrapper.modules():
                if isinstance(module, nn.LayerNorm):
                    for p in module.parameters():
                        p.requires_grad_(True)

    def forward(self, x, x_texture=None):
        x_texture = x if x_texture is None else x_texture
        x1 = self.backbone(x)
        x2 = self.backbone2(x_texture)
        out_list = list()
        for _x1, _x2, p1, p2, o, linear in zip(x1, x2, self.reproject1, self.reproject2, self.out_conv, self.align_linear):
            _x1 = p1(_x1)
            _x2 = p2(_x2)
            cat = torch.cat([_x1, _x2], dim=1)
            out = o(cat)
            out_list.append(linear(self.pool(out)))
        x = torch.cat(out_list, dim=1)

        color_result, color_logvar = self.color_predict_head(x)
        texture_result, texture_logvar = self.texture_predict_head(x)
        return {
            "color": color_result, "texture": texture_result,
            "color_logvar": color_logvar, "texture_logvar": texture_logvar
        }


class GaussianScoreHead(nn.Module):
    """Bounded score mean and learned log variance."""
    def __init__(self, in_channels, lo=1.0, hi=10.0):
        super().__init__()
        hidden = max(32, in_channels // 4)
        self.mean = nn.Sequential(nn.Linear(in_channels, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.logvar = nn.Sequential(nn.Linear(in_channels, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.logvar[-1].weight)
        nn.init.constant_(self.logvar[-1].bias, -1.5)
        self.lo, self.hi = lo, hi

    def forward(self, x):
        mu = self.lo + (self.hi - self.lo) * torch.sigmoid(self.mean(x))
        return mu, self.logvar(x)

class JadeQualityPredictor(nn.Module):
    def __init__(self, model_info):
        super().__init__()
        self.backbone = BackBone(**model_info)
        self.pool = create_pooling_layers() if model_info["pool"] else nn.Identity()
        self.color_predict_head = PredictHead(model_info["out_channels"][-1])
        self.texture_predict_head = PredictHead(model_info["out_channels"][-1])

    def forward(self, x):
        x = self.backbone(x)
        x = [self.pool(_x) for _x in x]
        color_result = self.color_predict_head(x[-1])
        texture_result = self.texture_predict_head(x[-1])
        return {
            "color": color_result,
            "texture": texture_result
        }
