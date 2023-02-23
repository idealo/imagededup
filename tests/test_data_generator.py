from pathlib import Path, PurePath
from typing import List, Tuple

import torch

from imagededup.methods import CNN
from imagededup.utils.data_generator import img_dataloader

p = Path(__file__)
IMAGE_DIR = p.parent / 'data/base_images'
FORMATS_IMAGE_DIR = p.parent / 'data/formats_images'
NESTED_IMAGE_DIR = p.parent / 'data/mixed_nested_images'

TEST_BATCH_SIZE = 3


def _init_dataloader(imdir: PurePath, recursive: bool) -> torch.utils.data.DataLoader:
    cnn = CNN()
    dataloader = img_dataloader(
        image_dir=imdir,
        batch_size=TEST_BATCH_SIZE,
        basenet_preprocess=cnn.apply_preprocess,
        recursive=recursive,
        num_workers=0
    )
    return dataloader


def _iterate_over_dataloader(dataloader: torch.utils.data.DataLoader) -> Tuple[List, List, List]:
    all_filenames, ims_arr, all_bad_images = [], [], []

    for ims, filenames, bad_images in dataloader:
        ims_arr.extend(ims)
        all_filenames.extend(filenames)
        all_bad_images.extend(bad_images)

    return all_filenames, ims_arr, all_bad_images


def test__data_generator():
    dataloader = _init_dataloader(imdir=IMAGE_DIR, recursive=False)
    all_filenames, ims_arr, all_bad_images = _iterate_over_dataloader(dataloader)
    all_ims = torch.stack(ims_arr)
    assert all_ims.shape == tuple([10, 3, 224, 224])
    assert len(all_filenames) == 10  # 10 images in the directory
    assert len(all_bad_images) == 0


def test_recursive_true_and_corrupt_file_ignored():
    dataloader = _init_dataloader(imdir=NESTED_IMAGE_DIR, recursive=True)
    all_filenames, ims_arr, all_bad_images = _iterate_over_dataloader(dataloader)
    all_ims = torch.stack(ims_arr)
    assert all_ims.shape == tuple([5, 3, 224, 224])
    assert len(all_filenames) == 5
    assert len(all_bad_images) == 1


def test_recursive_disabled_by_default():
    dataloader = _init_dataloader(imdir=NESTED_IMAGE_DIR, recursive=False)
    all_filenames, ims_arr, all_bad_images = _iterate_over_dataloader(dataloader)
    all_ims = torch.stack(ims_arr)
    assert all_ims.shape == tuple([1, 3, 224, 224])
    assert len(all_filenames) == 1
    assert len(all_bad_images) == 1
