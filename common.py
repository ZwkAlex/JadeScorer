from typing import Union

import torch
from torch import nn
from torch.nn.modules.utils import _pair


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, kernel_size=1, stride=1, padding=None, group=1, act: Union[bool, nn.Module] = True):
        # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=_pair(kernel_size),
            stride=_pair(stride),
            padding=autopad(kernel_size, padding),
            groups=group,
            bias=False
        )
        self.bn = nn.BatchNorm2d(c2, eps=1e-5, momentum=0.1)
        self.act = nn.SiLU() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ELAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.conv2 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.conv3 = nn.Sequential(
            Conv(hidden_channels, hidden_channels, kernel_size=3, stride=1),
            Conv(hidden_channels, hidden_channels, kernel_size=3, stride=1),
        )
        self.conv4 = nn.Sequential(
            Conv(hidden_channels, hidden_channels, kernel_size=3, stride=1),
            Conv(hidden_channels, hidden_channels, kernel_size=3, stride=1),
        )
        self.out_conv = Conv(hidden_channels * 4, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.out_conv(x)
        return x


class MPConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mp = MP()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.conv2 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.conv3 = Conv(hidden_channels, hidden_channels, kernel_size=3, stride=2)

    def forward(self, x):
        x1 = self.conv1(self.mp(x))
        x2 = self.conv3(self.conv2(x))

        x = torch.cat([x1, x2], dim=1)
        return x


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


