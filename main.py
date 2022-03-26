# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : main.py
# @Software: PyCharm
from engines.utils.logger import get_logger
from configure import Configure
from engines.train import train
from engines.model import Model
from engines.predict import Predictor
import argparse
import os
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entity extractor by binary tagging')
    parser.add_argument('--config_file', default='system.config', help='Configuration File')
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    configs = Configure(config_file=args.config_file)
    logger = get_logger(configs.log_dir)
    configs.show_data_summary(logger)
    mode = configs.mode.lower()
    if mode == 'train':
        logger.info('mode: train')
        train(configs, device, logger)
    elif mode == 'interactive_predict':
        logger.info('mode: predict_one')
        num_labels = len(configs.class_name)
        model = Model(hidden_size=768, num_labels=num_labels).to(device)
        model.load_state_dict(torch.load(os.path.join(configs.checkpoints_dir, 'best_model.pkl')))
        model.eval()
        predictor = Predictor(configs, device, logger)
        while True:
            logger.info('please input a sentence (enter [exit] to exit.)')
            sentence = input()
            if sentence == 'exit':
                break
            result = predictor.predict_one(sentence, model)
            print(result)
