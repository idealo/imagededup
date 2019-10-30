# Encoding generation
It might be desirable to only generate the hashes/cnn encodings for a given image or all images in a directory instead
of directly deduplicating using find_duplicates method. Encodings can be generated for a directory of images or for a 
single image:

- [Encoding generation for all images in a directory](#encoding-generation-for-all-images-in-a-directory)
- [Encoding generation for a single image](#encoding-generation-for-a-single-image)


## Encoding generation for all images in a directory
To generate encodings for all images in an image directory *encode_images* function can be used. The general api for 
using *encode_images* is:
```python
from imagededup.methods import <method-name>
method_object = <method-name>()
encodings = method_object.encode_images(image_dir='path/to/image/directory')
```
where the returned variable *encodings* is a dictionary mapping image file names to corresponding encoding:
```
{
  'image1.jpg': <encoding-image-1>,
  'image2.jpg': <encoding-image-2>,
   ..
}
```
For hashing algorithms, the encodings are 64 bit hashes represented as 16 character hexadecimal strings.

For cnn, the encodings are numpy array with shape (1, 1024).

The 'method-name' corresponds to one of the deduplication methods available and can be set to:

- PHash
- AHash
- DHash
- WHash
- CNN


#### Options
- image_dir: Path to the image directory for which encodings are to be generated.

#### Considerations

- If an image in the image directory can't be loaded, no encodings are generated for the image. Hence, there is no entry 
for the image in the returned encodings dictionary.
- Supported image formats: 'JPEG', 'PNG', 'BMP', 'MPO', 'PPM', 'TIFF', 'GIF', 'SVG', 'PGM', 'PBM'.

#### Examples

Generating encodings using Difference hash:
```python
from imagededup.methods import DHash
dhasher = DHash()
encodings = dhasher.encode_images(image_dir='path/to/image/directory')
```

## Encoding generation for a single image
To generate encodings for a single image *encode_image* function can be used. The general api for 
using *encode_image* is:
```python
from imagededup.methods import <method-name>
method_object = <method-name>()
encoding = method_object.encode_image(image_file='path/to/image/file')
```
where the returned variable *encoding* is either a hexadecimal string if a hashing method is used or a (1, 1024) numpy 
array if cnn is used.

#### Options
- image_file: Optional, path to the image file for which encodings are to be generated.
- image_array: Optional, used instead of *image_file* attribute. A numpy array representing the image.

#### Considerations

- If the image can't be loaded, no encodings are generated for the image and *None* is returned.
- Supported image formats: 'JPEG', 'PNG', 'BMP', 'MPO', 'PPM', 'TIFF', 'GIF', 'SVG', 'PGM', 'PBM'.

#### Examples

Generating encodings using Difference hash:
```python
from imagededup.methods import DHash
dhasher = DHash()
encoding = dhasher.encode_image(image_file='path/to/image/file')
```