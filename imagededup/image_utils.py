import os
from pathlib import PosixPath


"""
Validation of directory images
For all: Restrict image sizes to be greater than a certain size
Increase image input acceptance type for single images(jpg, bmp, png, jpeg, etc.)
? Allow acceptance of os.path in addition to already existing Path and numpy image array

"""


def check_valid_file(path_image: PosixPath) -> int:
    if not os.path.exists(path_image):
        raise FileNotFoundError('Ensure that the file exists!')
    str_name = path_image.name
    if not (str_name.endswith('.jpeg') or str_name.endswith('.jpg') or str_name.endswith('.bmp') or
            str_name.endswith('.png')):
        raise TypeError('Image formats supported: .jpg, .jpeg, .bmp, .png')
    return 1


def check_directory_files(path_image: PosixPath):
    """check if each image in a directory is a valid image
    return 1 if all valid
    return 0 if files can't be loaded. Also log info about missing files and continue with feature generation
    if there is at least 1 valid file in the directory"""
    return 0


def check_directory_files(path_dir: PosixPath):
    """Checks if all files in *image_dir* are valid images.
    Checks if files are images with extention 'JPEG' or 'PNG' and if they are not truncated.
    self.logger.infos filenames that didn't pass the validation.
    Sets:
        self.valid_image_ids: a list of valid image.
    """
    self.logger.info('\n****** Running image validation ******\n')

    # validate images, use multiprocessing
    files = [str(i.absolute()) for i in path_dir.glob('*')]
    files.sort()

    results = parallelise(validate_image, files)

    valid_image_files = [j for i, j in enumerate(files) if results[i][0]]
    self.valid_image_ids = [Path(i).name for i in valid_image_files]

    # return list of invalid images to user and save them if there are more than 10
    invalid_image_files = [
        (j, str(results[i][1])) for i, j in enumerate(files) if not results[i][0]
    ]

    if invalid_image_files:
        self.logger.info('The following files are not valid image files:')
        for file_name, error_msg in invalid_image_files[:10]:
            self.logger.info('- {} ({})'.format(file_name, error_msg))
        if len(invalid_image_files) > 10:
            save_json(invalid_image_files, self.job_dir / 'invalid_image_files.json')
            self.logger.info(
                (
                    'NOTE: More than 10 files were identified as invalid image files.\n'
                    'The full list of those files has been saved here:\n{}'.format(
                        self.job_dir / 'invalid_image_files.json'
                    )
                )
            )
