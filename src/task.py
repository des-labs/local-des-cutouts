import logging
import sys
import yaml
import os
from astropy.io import fits
import subprocess

logger = logging.getLogger(__name__)


def execute_task(config):
    # Dump cutout config to YAML file in working directory
    cutout_config_file = 'cutout_config.yaml'
    with open(cutout_config_file, 'w') as file:
        yaml.dump(config['spec'], file)

    # Launch the cutout generation using the Message Passing Interface (MPI) system to parallelize execution
    num_cpus = config['spec']['num_cpus'] if 'num_cpus' in config['spec'] and isinstance(
        config['spec']['num_cpus'], int) else 1
    args = 'mpirun -n {} python3 bulkthumbs.py --config {}'.format(
        num_cpus, cutout_config_file)
    try:
        run_output = subprocess.check_output([args], shell=True)
        logger.info(run_output)
    except subprocess.CalledProcessError as e:
        logger.error(e.output)
        raise

    # Verifying outputs
    path = config['spec']['outdir']
    for file in os.listdir(path):
        if file.endswith(".fits"):
            try:
                fullpath = os.path.join(path, file)
                hdus = fits.open(fullpath, checksum=True)
                hdus.verify()
            except Exception as err:
                logger.error(err)
                raise


def run(config, user_dir='/home/worker/output'):

    # Make the cutout subdirectory if it does not already exist.
    cutout_dir = os.path.join(user_dir, 'cutouts')
    os.makedirs(cutout_dir, exist_ok=True)
    config['cutout_dir'] = cutout_dir

    # Configure logging
    formatter = logging.Formatter(
        "%(asctime)s  %(name)8s  %(levelname)5s  %(message)s")
    fh = logging.FileHandler(config['spec']['log'])
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    # Execute the primary task
    execute_task(config)


if __name__ == "__main__":

    # Import job configuration
    try:
        input_file = sys.argv[1]
    except Exception:
        input_file = 'input/cutout_config.yaml'
    with open(input_file) as cfile:
        config = yaml.safe_load(cfile)

    with open('input/positions.csv', 'r') as csvfile:
        config['spec']['positions'] = csvfile.read()

    run(config)
