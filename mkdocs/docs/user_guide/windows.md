# Windows
The following code snippet on Windows is likely to break with a `RuntimeError`:

```python
from imagededup.methods import PHash
phasher = PHash()

# Generate encodings for all images in an image directory
encodings = phasher.encode_images(image_dir='path/to/image/directory')

# Find duplicates using the generated encodings
duplicates = phasher.find_duplicates(encoding_map=encodings)

# plot duplicates obtained for a given file using the duplicates dictionary
from imagededup.utils import plot_duplicates
plot_duplicates(image_dir='path/to/image/directory',
                duplicate_map=duplicates,
                filename='ukbench00120.jpg')
```

```
RuntimeError:
    An attempt has been made to start a new process before the
    current process has finished its bootstrapping phase.

    This probably means that you are not using fork to start your
    child processes and you have forgotten to use the proper idiom
    in the main module:

        if __name__ == '__main__':
            freeze_support()
            ...

    The "freeze_support()" line can be omitted if the program
    is not going to be frozen to produce an executable.
```
The error occurs due to the underlying dependency `multiprocessing` which observes certain restrictions when working on Windows as explained [here](https://docs.python.org/2/library/multiprocessing.html#windows).

## The Fix

Just enclose the logic in a `if __name__ == '__main__':` entry point:

```python
from imagededup.methods import PHash

if __name__ == '__main__':
    phasher = PHash()

    # Generate encodings for all images in an image directory
    encodings = phasher.encode_images(image_dir='path/to/image/directory')

    # Find duplicates using the generated encodings
    duplicates = phasher.find_duplicates(encoding_map=encodings)

    # plot duplicates obtained for a given file using the duplicates dictionary
    from imagededup.utils import plot_duplicates
    plot_duplicates(image_dir='path/to/image/directory',
                    duplicate_map=duplicates,
                    filename='ukbench00120.jpg')
```
Also see issues [194](https://github.com/idealo/imagededup/issues/194) and [214](https://github.com/idealo/imagededup/issues/214).