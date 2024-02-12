import click
import PROT
import yaml
import os

from PROT import main_multitask
from PROT.utils import setup_logging


@click.group()
def cli():
    """ CLI for PROT """
    pass


@cli.command()
@click.option(
    '-c',
    '--config-filename',
    default=['configs/esm2_1GPU.yml'],
    multiple=True,
    help=(
        'Path to training configuration file. If multiple are provided, runs will be '
        'executed in order'
    )
)
@click.option('-r', '--resume', default=None, type=str, help='path to checkpoint')
def train(config_filename: str, resume: str):
    """ Entry point to start training run(s). """
    for c in config_filename:
        with open(c) as fh:
            config = yaml.safe_load(fh)
        setup_logging(config)
        main_multitask.train(config, resume)


@cli.command()
@click.option('-c', '--config-filename', default="configs/esm2_1GPU.yml", type=str, help='Path to model configuration file.')
@click.option('-d', '--model_data', default='weights/model_best.pth', type=str, help='Path to model data')
@click.option('-i', '--input_data', default='exemple.fasta', type=str, help='Path to input data')
@click.option('-p', '--pred_name', default="SecondaryFeatures", type=str, help='Name of the prediction class')
@click.option('-v', '--vizualization', default=False, type=bool, help='Plot the vizualization plot')
def predict(config_filename: str, pred_name: str, model_data: str, input_data: str, vizualization: bool):
    with open(config_filename) as fh:
        config = yaml.safe_load(fh)
    main_multitask.predict(config, pred_name, model_data, input_data, vizualization)


@cli.command()
@click.option(
    '-c',
    '--config-filename',
    default=['configs/esm2_1GPU.yml'],
    multiple=True,
    help=(
        'Path to training configuration file. If multiple are provided, runs will be '
        'executed in order'
    )
)
@click.option('-r', '--resume', default=None, type=str, help='path to checkpoint')
def load_config(filename: str) -> dict:
    """ Load a configuration file as YAML. """
    with open(filename) as fh:
        config = yaml.safe_load(fh)
    return config
