from imagededup.methods import cnn
from pathlib import Path
from PIL import Image
from pathlib import PosixPath
import pytest
import numpy as np


@pytest.fixture(scope='module')
def initialized_cnn_obj():
    cnn_obj = cnn.CNN()
    return cnn_obj
