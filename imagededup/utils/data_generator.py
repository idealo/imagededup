from pathlib import PurePath
from typing import Tuple, List, Callable, Optional

import numpy as np
import torch
from PIL.Image import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import transforms

from imagededup.utils.image_utils import load_image
from imagededup.utils.general_utils import generate_files


class ImgDataset(Dataset):
    def __init__(self, image_dir: PurePath, target_size: Tuple[int, int], recursive: Optional[bool]) -> None:
        self.image_dir = image_dir
        self.target_size = target_size
        self.recursive = recursive

        self.transform = transforms.Compose([transforms.Resize(target_size),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

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


def img_dataloader(image_dir: PurePath, batch_size: int, target_size: Tuple[int, int], recursive: Optional[bool]):
    img_dataset = ImgDataset(image_dir=image_dir, target_size=target_size, recursive=recursive)
    return DataLoader(dataset=img_dataset, batch_size=batch_size, collate_fn=_collate_fn)


class MobilenetV3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(pretrained=True).eval()
        self.mobilenet_gap_op = torch.nn.Sequential(mobilenet.features, mobilenet.avgpool)

    def forward(self, x):
        return self.mobilenet_gap_op(x)


def generate_features(dataloader, model):
    feat_arr = []
    all_filenames = []

    for ims, filenames, bad_images in dataloader:
        arr = model(ims)
        feat_arr.extend(arr[:2])
        all_filenames.extend(filenames)
    
    if len(bad_images):
        print('Found some bad images, ignoring for encoding generation ..')

    feat_arr = torch.stack(feat_arr).squeeze()
    valid_filenames = [filename for filename in all_filenames if filename]
    return feat_arr, valid_filenames


class DataGenerator(Sequence):
    """Class inherits from Keras Sequence base object, allows to use multiprocessing in .fit_generator.

    Attributes:
        image_dir: Path of image directory.
        batch_size: Number of images per batch.
        basenet_preprocess: Basenet specific preprocessing function.
        target_size: Dimensions that images get resized into when loaded.
        recursive: Optional, find images recursively in the image directory.
    """

    def __init__(
        self,
        image_dir: PurePath,
        batch_size: int,
        basenet_preprocess: Callable,
        target_size: Tuple[int, int],
        recursive: Optional[bool] = False,
    ) -> None:
        """Init DataGenerator object.
        """
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.basenet_preprocess = basenet_preprocess
        self.target_size = target_size
        self.recursive = recursive

        self._get_image_files()
        self.indexes = np.arange(len(self.image_files))
        self.valid_image_files = self.image_files

    def _get_image_files(self) -> None:
        self.image_files = sorted(
            generate_files(self.image_dir, self.recursive)
        )  # ignore hidden files

    def __len__(self) -> int:
        """Number of batches in the Sequence."""
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        """Get batch at position `index`.
        """
        batch_indexes = self.indexes[
                        index * self.batch_size: (index + 1) * self.batch_size
                        ]
        batch_samples = [self.image_files[i] for i in batch_indexes]
        X = self._data_generator(batch_samples)
        return X

    def _data_generator(
            self, image_files: List[PurePath]
    ) -> Tuple[np.array, np.array]:
        """Generate data from samples in specified batch."""
        #  initialize images and labels tensors for faster processing
        X = np.empty((len(image_files), *self.target_size, 3))

        invalid_image_idx = []
        for i, image_file in enumerate(image_files):
            # load and randomly augment image
            img = load_image(
                image_file=image_file, target_size=self.target_size, grayscale=False
            )

            if img is not None:
                X[i, :] = img

            else:
                invalid_image_idx.append(i)
                self.valid_image_files = [_file for _file in self.valid_image_files if _file != image_file]

        if invalid_image_idx:
            X = np.delete(X, invalid_image_idx, axis=0)

        # apply basenet specific preprocessing
        # input is 4D numpy array of RGB values within [0, 255]
        X = self.basenet_preprocess(X)

        return X