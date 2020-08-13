from utils.logger import get_logger
from configure import Configure
from engines.train import train
import argparse
import os


def fold_check(configures):
    checkpoints_dir = 'checkpoints_dir'
    if not os.path.exists(configures.checkpoints_dir) or not hasattr(configures, checkpoints_dir):
        print('checkpoints fold not found, creating...')
        dir_name = configures.checkpoints_dir.split('/')[0]
        os.mkdir(dir_name)

    log_dir = 'log_dir'
    if not os.path.exists(configures.log_dir) or not hasattr(configures, log_dir):
        print('log fold not found, creating...')
        os.mkdir(configures.log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entity extractor by binary tagging')
    parser.add_argument('--config_file', default='system.config', help='Configuration File')
    args = parser.parse_args()
    configs = Configure(config_file=args.config_file)
    fold_check(configs)
    logger = get_logger(configs.log_dir)
    configs.show_data_summary(logger)
    mode = configs.mode.lower()
    if mode == 'train':
        logger.info('mode: train')
        train(configs, logger)
