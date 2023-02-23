from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import transforms
from torchvision.models import vit_b_16
from torchvision.models.vision_transformer import ViT_B_16_Weights

DEFAULT_MODEL_NAME = 'default_model'
custom_model = namedtuple('custom_model', 'model transform name', defaults=[None, None, DEFAULT_MODEL_NAME])


class MobilenetV3(torch.nn.Module):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    def __init__(self) -> None:
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1').eval()
        self.mobilenet_gap_op = torch.nn.Sequential(
            mobilenet.features, mobilenet.avgpool
        )

    def forward(self, x) -> torch.tensor:
        x = self.mobilenet_gap_op(x)
        x = x.squeeze(dim=3).squeeze(dim=2)
        return x


class ViT(torch.nn.Module):
    transform = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()

    def __init__(self) -> None:
        super().__init__()
        self.model = vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1').eval()
        self.hidden_dim = 768
        self.patch_size = 16
        self.image_size = 384
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.model.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x) -> torch.tensor:
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.model.encoder(x)

        # mean of all encoder outputs, performs better than the classifier "token" as used by standard language architectures
        x = torch.mean(x, 1)
        return x
