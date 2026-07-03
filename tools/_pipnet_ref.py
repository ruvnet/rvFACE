"""PIPNet ResNet-18 reference architecture (xlite-dev/torchlm, MIT).

Importable module (no side effects on import): defines :class:`PIPNet`, a stock
torchvision ResNet-18 stem+trunk with five parallel 1x1 conv heads, whose
``state_dict`` keys match the released
``pipnet_resnet18_10x68x32x256_300w.pth`` verbatim (``conv1``, ``bn1``,
``layer1``..``layer4`` + ``cls_layer``/``x_layer``/``y_layer``/``nb_x_layer``/
``nb_y_layer``).

Used by:
- ``tools/fetch_and_convert.py`` to validate the converted safetensors
  (``load_state_dict(..., strict=True)``);
- ``tools/gen_fixtures.py`` to run the real weights on a seeded input and dump
  the ``landmark-pipnet-real`` parity fixture.

Forward returns the five raw score maps (no sigmoid); for a 256x256 input at
net stride 32 they are cls/x/y ``[1,68,8,8]`` and nb_x/nb_y ``[1,680,8,8]``.
The heatmap+offset+neighbor-regression (NRM) decode that turns these into 68
landmark coordinates lives in the Rust port, not here.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
import torchvision

NUM_LMS = 68
NUM_NB = 10
INPUT_SIZE = 256
NET_STRIDE = 32


class PIPNet(nn.Module):
    def __init__(self, num_lms: int = NUM_LMS, num_nb: int = NUM_NB):
        super().__init__()
        rn = torchvision.models.resnet18(weights=None)
        self.conv1 = rn.conv1
        self.bn1 = rn.bn1
        self.maxpool = rn.maxpool
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4
        self.cls_layer = nn.Conv2d(512, num_lms, 1)
        self.x_layer = nn.Conv2d(512, num_lms, 1)
        self.y_layer = nn.Conv2d(512, num_lms, 1)
        self.nb_x_layer = nn.Conv2d(512, num_lms * num_nb, 1)
        self.nb_y_layer = nn.Conv2d(512, num_lms * num_nb, 1)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return (
            self.cls_layer(x),
            self.x_layer(x),
            self.y_layer(x),
            self.nb_x_layer(x),
            self.nb_y_layer(x),
        )


def build_pipnet() -> PIPNet:
    """A fresh PIPNet in eval mode (random init; caller loads real weights)."""
    return PIPNet().eval()
