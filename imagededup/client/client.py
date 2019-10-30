import importlib
import click

from typing import Optional
from pathlib import PosixPath


@click.group()
def cli():
    pass


@cli.command()
@click.option('--image_dir', help='Path to the directory containing all the images.', type=str, required=True)
@click.option('--method', help='Select which algorithm to use.',
              type=click.Choice(['PHash', 'DHash', 'WHash', 'AHash', 'CNN']), required=True)
@click.option('--outfile', help='Name of the file the results should be written to.', type=str)
@click.option('--min_similarity_threshold',
              help='For CNN only: threshold value (must be float between -1.0 and 1.0). Default is 0.9.',
              type=click.FloatRange(-1.0, 1.0),
              default=0.9)
@click.option('--max_distance_threshold',
              help='For hashing methods only: threshold value (must be integer between 0 and 64). Default is 10.',
              type=click.IntRange(0, 64), default=10)
@click.option('--scores',
              help='Boolean indicating whether scores are to be returned along with retrieved duplicates.',
              type=bool)
def find_duplicates(image_dir: PosixPath,
                    method: str,
                    outfile: Optional[str],
                    min_similarity_threshold: float,
                    max_distance_threshold: int,
                    scores: bool) -> None:
    selected_method = getattr(importlib.import_module('imagededup.methods'), method)()
    encodings = selected_method.encode_images(image_dir)

    if method == 'CNN':
        duplicates = selected_method.find_duplicates(encoding_map=encodings,
                                                     outfile=outfile,
                                                     min_similarity_threshold=min_similarity_threshold,
                                                     scores=scores)
    else:
        duplicates = selected_method.find_duplicates(encoding_map=encodings,
                                                     outfile=outfile,
                                                     max_distance_threshold=max_distance_threshold,
                                                     scores=scores)
    if outfile is None:
        click.echo(duplicates)
