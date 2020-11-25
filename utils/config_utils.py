import os
import yaml
import pathlib
import logging


logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)


def parse(config_path):
    if not config_path.exists():
        logger.info("config file not exist...")

    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

default_cfg_path = pathlib.Path(__file__).parent.absolute() / pathlib.PurePosixPath("../configs/summary-template.yaml")
default_cfg = parse(default_cfg_path)
