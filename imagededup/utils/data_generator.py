from pathlib import PurePath
from typing import Dict, Callable, Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from imagededup.utils.image_utils import load_image
from imagededup.utils.general_utils import generate_files


class ImgDataset(Dataset):
    def __init__(
        self,
        image_dir: PurePath,
        basenet_preprocess: Callable[[np.array], torch.tensor],
        recursive: Optional[bool],
    ) -> None:
        self.image_dir = image_dir
        self.basenet_preprocess = basenet_preprocess
        self.recursive = recursive
        self.image_files = sorted(
            generate_files(self.image_dir, self.recursive)
        )  # ignore hidden files

    def __len__(self) -> int:
        """Number of images."""
        return len(self.image_files)

    def __getitem__(self, item) -> Dict:
        im_arr = load_image(self.image_files[item], target_size=None, grayscale=None)
        if im_arr is not None:
            img = self.basenet_preprocess(im_arr)
            return {'image': img, 'filename': self.image_files[item]}
        else:
            return {'image': None, 'filename': self.image_files[item]}


def _collate_fn(batch: List[Dict]) -> Tuple[torch.tensor, str, str]:
    ims, filenames, bad_images = [], [], []

    for b in batch:
        im = b['image']
        if im is not None:
            ims.append(im)
            filenames.append(b['filename'])
        else:
            bad_images.append(b['filename'])
    return torch.stack(ims), filenames, bad_images


def img_dataloader(
    image_dir: PurePath,
    batch_size: int,
    basenet_preprocess: Callable[[np.array], torch.tensor],
    recursive: Optional[bool],
    num_workers: int
) -> DataLoader:
    img_dataset = ImgDataset(
        image_dir=image_dir, basenet_preprocess=basenet_preprocess, recursive=recursive
    )
    return DataLoader(
        dataset=img_dataset, batch_size=batch_size, collate_fn=_collate_fn, num_workers=num_workers
    )


class MobilenetV3(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        mobilenet = models.mobilenet_v3_small(pretrained=True).eval()
        self.mobilenet_gap_op = torch.nn.Sequential(
            mobilenet.features, mobilenet.avgpool
        )

    def forward(self, x) -> torch.tensor:
        return self.mobilenet_gap_op(x)


from torchvision.models import vit_b_16
from torchvision.models.vision_transformer import ViT_B_16_Weights
import torch
import torch.nn as nn


class Mynet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1').eval()
        self.hidden_dim = 768
        self.patch_size = 16
        self.image_size = 384
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.transform = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()

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

        # Classifier "token" as used by standard language architectures
        # x = x[:, 0]
        x = torch.mean(x, 1)
        # import pdb; pdb.set_trace()

        return x



