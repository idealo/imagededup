from pathlib import PurePath
from typing import Tuple, List, Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import transforms

from imagededup.utils.image_utils import load_image
from imagededup.utils.general_utils import generate_files


class ImgDataset(Dataset):
    def __init__(
        self,
        image_dir: PurePath,
        target_size: Tuple[int, int],
        recursive: Optional[bool],
    ) -> None:
        self.image_dir = image_dir
        self.target_size = target_size
        self.recursive = recursive

        self.transform = transforms.Compose(
            [
                transforms.Resize(target_size),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.image_files = sorted(
            generate_files(self.image_dir, self.recursive)
        )  # ignore hidden files

    def __len__(self) -> int:
        """Number of images."""
        return len(self.image_files)

    def __getitem__(self, item):
        try:
            img = Image.open(self.image_files[item])
            if img.mode != 'RGB':
                # convert to RGBA first to avoid warning
                # we ignore alpha channel if available
                img = img.convert('RGBA').convert('RGB')
            img = self.transform(img)
        except:
            return {'image': None, 'filename': self.image_files[item]}
        return {'image': img, 'filename': self.image_files[item]}


def _collate_fn(batch):
    ims = []
    filenames = []
    bad_images = []

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
    target_size: Tuple[int, int],
    recursive: Optional[bool],
):
    img_dataset = ImgDataset(
        image_dir=image_dir, target_size=target_size, recursive=recursive
    )
    return DataLoader(
        dataset=img_dataset, batch_size=batch_size, collate_fn=_collate_fn
    )


class MobilenetV3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(pretrained=True).eval()
        self.mobilenet_gap_op = torch.nn.Sequential(
            mobilenet.features, mobilenet.avgpool
        )

    def forward(self, x):
        return self.mobilenet_gap_op(x)

