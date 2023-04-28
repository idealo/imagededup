# Using custom models for CNN

To allow users to use custom models for encoding generation, we provide a `CustomModel` construct which serves as a wrapper for a user-defined feature extractor. The `CustomModel` consists of the following attributes:

- `name`: The name of the custom model. Can be set to any string.
- `model`: A PyTorch model object, which is a subclass of `torch.nn.Module` and implements the `forward` method. The output of the forward method should be a tensor of shape (batch_size x features) . Alternatively, a `__call__` method is also accepted.
- `transform`: A function that transforms a `PIL.Image` object into a PyTorch tensor. Should correspond to the preprocessing logic of the supplied model.


`CustomModel` is provided while initializing the `cnn` object and can be used in the following 2 scenarios:

1. Using the models provided with the `imagededup` package.
There are 3 models provided currently:
    - `MobileNetV3` ([MobileNetV3 Small](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html#torchvision.models.mobilenet_v3_small))- This is the default.
    - `ViT` ([Vision Transformer- B16 IMAGENET1K_SWAG_E2E_V1](https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html?highlight=vit_b_16#torchvision.models.vit_b_16))    
    - `EfficientNet` ([EfficientNet B4- IMAGENET1K_V1](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b4.html?highlight=efficientnet_b4_weights#torchvision.models.EfficientNet_B4_Weights))

```python
from imagededup.methods import CNN

# Get CustomModel construct
from imagededup.utils import CustomModel

# Get the prepackaged models from imagededup
from imagededup.utils.models import ViT, MobilenetV3, EfficientNet


# Declare a custom config with CustomModel, the prepackaged models come with a name and transform function
custom_config = CustomModel(name=EfficientNet.name,
                            model=EfficientNet(), 
                            transform=EfficientNet.transform)

# Use model_config argument to pass the custom config
cnn = CNN(model_config=custom_config)

# Use the model as usual
...

```

2.Using a user-defined custom model.
```python
from imagededup.methods import CNN

# Get CustomModel construct
from imagededup.utils import CustomModel

# Import necessary pytorch constructs for initializing a custom feature extractor
import torch
from torchvision.transforms import transforms

# Declare custom feature extractor class
class MyModel(torch.nn.Module):
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    name = 'my_custom_model'

    def __init__(self):
        super().__init__()
        # Define the layers of the model here

    def forward(self, x):
        # Do something with x
        return x

custom_config = CustomModel(name=MyModel.name,
                            model=MyModel(),
                            transform=MyModel.transform)

cnn = CNN(model_config=custom_config)

# Use the model as usual
...

```
It is not necessary to bundle `name` and `transform` functions with the `model` class. They can be passed separately as well.


Examples for both scenarios can be found in the [examples section](https://github.com/idealo/imagededup/tree/master/examples).