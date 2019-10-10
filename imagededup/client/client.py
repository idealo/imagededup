import click

from typing import Optional
from pathlib import PosixPath


@click.group()
def cli():
    pass


@cli.command()
@click.option('--image_dir', help='Path to the directory containing all the images.', required=True, type=str)
@click.option('--method', help='Select which algorithm to use.', required=True,
              type=click.Choice(['PHash', 'DHash', 'WHash', 'AHash', 'CNN']))
@click.option('--outfile', help='Name of the file the results should be written to.', type=str)
@click.option('--max_distance_threshold', default=10,
              help='Hamming distance between two images below which retrieved duplicates are valid.', type=int)
@click.option('--scores',
              help='Boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.',
              type=bool)
def find_duplicates(image_dir: PosixPath,
                   method: str,
                   outfile: Optional[str],
                   max_distance_threshold: int,
                   scores: bool) -> None:
    import imagededup.methods
    selected_method = eval('imagededup.methods.{}()'.format(method))
    encodings = selected_method.encode_images(image_dir)
    duplicates = selected_method.find_duplicates(encoding_map=encodings,
                                                 outfile=outfile,
                                                 max_distance_threshold=max_distance_threshold,
                                                 scores=scores)
    if outfile is None:
        click.echo(duplicates)
