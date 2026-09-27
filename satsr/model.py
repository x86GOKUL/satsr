"""
RRDBNet (Real-ESRGAN backbone) and weight loading for satsr.

Pure PyTorch (no basicsr). Supports 3-channel (RGB) and 4-channel (RGB+NIR)
variants; when loading 3-ch pretrained weights into a 4-ch model, the extra
input/output slot is initialised from the red channel.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _make_layer(block, n, **kw):
    return nn.Sequential(*[block(**kw) for _ in range(n)])


class ResidualDenseBlock(nn.Module):
    # layer names (conv1..conv5) MUST match the trained checkpoints
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, nf=64, nb=23, gc=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.body = _make_layer(RRDB, nb, nf=nf, gc=gc)
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def build_model(in_channels: int = 4, weights: str | None = None,
                device: str = "cpu") -> RRDBNet:
    """Construct the model and load weights, adapting 3->4 channels if needed."""
    model = RRDBNet(in_ch=in_channels, out_ch=in_channels, nf=64, nb=23, gc=32, scale=4)
    if weights:
        state = torch.load(weights, map_location="cpu")
        state = state.get("params_ema", state.get("params", state))
        own = model.state_dict()
        # infer source channel count from conv_first
        src_in = state["conv_first.weight"].shape[1]
        if src_in == in_channels:
            model.load_state_dict(state, strict=True)
        elif src_in == 3 and in_channels == 4:
            for k, v in state.items():
                if k == "conv_first.weight":
                    w = own[k].clone(); w[:, :3] = v; w[:, 3:4] = v[:, 0:1]; own[k] = w
                elif k == "conv_last.weight":
                    w = own[k].clone(); w[:3] = v; w[3:4] = v[0:1]; own[k] = w
                elif k == "conv_last.bias":
                    b = own[k].clone(); b[:3] = v; b[3:4] = v[0:1]; own[k] = b
                elif k in own and own[k].shape == v.shape:
                    own[k] = v
            model.load_state_dict(own, strict=True)
        else:
            raise ValueError(f"Cannot map weights with {src_in} channels to a "
                             f"{in_channels}-channel model")
    return model.eval().to(device)


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device
