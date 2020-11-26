import os
import yaml
import pathlib
import logging


logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)


def parse_cfg(config_path):
    if not config_path.exists():
        logger.info("config file not exist...")
        return None

    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def get_default_summary_config():
    default_cfg_path = pathlib.Path(__file__).parent.absolute() / pathlib.PurePosixPath("../configs/summary-template.yaml")
    default_cfg = parse_cfg(default_cfg_path)
    return default_cfg
