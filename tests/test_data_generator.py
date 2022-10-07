import pytest
from pathlib import Path

# from tensorflow.keras.applications.mobilenet import preprocess_input

from imagededup.utils.data_generator import DataGenerator

p = Path(__file__)
IMAGE_DIR = p.parent / 'data/base_images'
FORMATS_IMAGE_DIR = p.parent / 'data/formats_images'
NESTED_IMAGE_DIR = p.parent / 'data/mixed_nested_images'

TEST_BATCH_SIZE = 3
TEST_TARGET_SIZE = (224, 224)


@pytest.fixture(autouse=True)
def run_before_tests():
    global generator
    generator = DataGenerator(
        image_dir=IMAGE_DIR,
        batch_size=TEST_BATCH_SIZE,
        basenet_preprocess=preprocess_input,
        target_size=TEST_TARGET_SIZE,
    )


def test__init():
    assert generator.image_dir == IMAGE_DIR
    assert generator.batch_size == TEST_BATCH_SIZE
    assert generator.target_size == TEST_TARGET_SIZE
    assert generator.basenet_preprocess == preprocess_input


def test__len():
    assert generator.__len__() == 4


def test__get_item(mocker):
    mocker.patch.object(DataGenerator, '_data_generator')

    generator.__getitem__(0)

    generator._data_generator.assert_called_with(
        [
            IMAGE_DIR / 'ukbench00120.jpg',
            IMAGE_DIR / 'ukbench01380.jpg',
            IMAGE_DIR / 'ukbench08976.jpg',
        ]
    )


def test__data_generator():
    image_files = [
        IMAGE_DIR / 'ukbench09348.jpg',
        IMAGE_DIR / 'ukbench09012.jpg',
        IMAGE_DIR / 'ukbench09380.jpg',
    ]

    result = generator._data_generator(image_files=image_files)

    assert result.shape == tuple([TEST_BATCH_SIZE, *TEST_TARGET_SIZE, 3])


def test_valid_image_files_1():
    assert generator.valid_image_files == sorted(
        [x for x in IMAGE_DIR.glob('*') if x.is_file()]
    )


def test_valid_image_files_2():
    expected = [
        FORMATS_IMAGE_DIR / 'baboon.pgm',
        FORMATS_IMAGE_DIR / 'copyleft.tiff',
        FORMATS_IMAGE_DIR / 'giphy.gif',
        FORMATS_IMAGE_DIR / 'Iggy.1024.ppm',
        FORMATS_IMAGE_DIR / 'marbles.pbm',
        FORMATS_IMAGE_DIR / 'mpo_image.MPO',
        FORMATS_IMAGE_DIR / 'ukbench09380.bmp',
        FORMATS_IMAGE_DIR / 'ukbench09380.jpeg',
        FORMATS_IMAGE_DIR / 'ukbench09380.png',
        FORMATS_IMAGE_DIR / 'ukbench09380.svg',
    ]

    generator = DataGenerator(
        image_dir=FORMATS_IMAGE_DIR,
        batch_size=TEST_BATCH_SIZE,
        basenet_preprocess=preprocess_input,
        target_size=TEST_TARGET_SIZE,
    )

    generator.__getitem__(0)
    generator.__getitem__(1)
    generator.__getitem__(2)

    # generator._update_valid_files()
    assert sorted(generator.valid_image_files, key=lambda x: str(x).lower()) == expected


def test_recursive_image_files():
    generator2 = DataGenerator(
        image_dir=NESTED_IMAGE_DIR,
        batch_size=TEST_BATCH_SIZE,
        basenet_preprocess=preprocess_input,
        target_size=TEST_TARGET_SIZE,
        recursive=True,
    )

    assert len(generator2.valid_image_files) == 6

    assert generator2.valid_image_files == sorted(
        [x for x in NESTED_IMAGE_DIR.glob('**/*') if x.is_file() and not x.name.startswith('.')]
    )


def test_recursive_disabled_by_default():
    generator = DataGenerator(
        image_dir=NESTED_IMAGE_DIR,
        batch_size=TEST_BATCH_SIZE,
        basenet_preprocess=preprocess_input,
        target_size=TEST_TARGET_SIZE,
    )

    assert len(generator.valid_image_files) == 2

    assert generator.valid_image_files == sorted(
        [x for x in NESTED_IMAGE_DIR.glob('*') if x.is_file() and not x.name.startswith('.')]
    )
