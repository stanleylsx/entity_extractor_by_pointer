# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : configure.py
# @Software: PyCharm
import sys


class Configure:
    def __init__(self, config_file='system.config'):
        config = self.config_file_to_dict(config_file)

        # Status:
        the_item = 'mode'
        if the_item in config:
            self.mode = config[the_item]

        # Datasets(Input/Output):
        the_item = 'datasets_fold'
        if the_item in config:
            self.datasets_fold = config[the_item]
        the_item = 'train_file'
        if the_item in config:
            self.train_file = config[the_item]
        the_item = 'dev_file'
        if the_item in config:
            self.dev_file = config[the_item]
        the_item = 'test_file'
        if the_item in config:
            self.test_file = config[the_item]

        the_item = 'checkpoints_dir'
        if the_item in config:
            self.checkpoints_dir = config[the_item]

        the_item = 'log_dir'
        if the_item in config:
            self.log_dir = config[the_item]

        # Labeling Scheme
        the_item = 'class_name'
        if the_item in config:
            self.class_name = config[the_item]

        the_item = 'decision_threshold'
        if the_item in config:
            self.decision_threshold = config[the_item]

        # Training Settings:
        the_item = 'is_early_stop'
        if the_item in config:
            self.is_early_stop = self.str2bool(config[the_item])
        the_item = 'patient'
        if the_item in config:
            self.patient = int(config[the_item])

        the_item = 'epoch'
        if the_item in config:
            self.epoch = int(config[the_item])
        the_item = 'batch_size'
        if the_item in config:
            self.batch_size = int(config[the_item])

        the_item = 'dropout'
        if the_item in config:
            self.dropout = float(config[the_item])
        the_item = 'learning_rate'
        if the_item in config:
            self.learning_rate = float(config[the_item])

    @staticmethod
    def config_file_to_dict(input_file):
        config = {}
        fins = open(input_file, 'r', encoding='utf-8').readlines()
        for line in fins:
            if len(line) > 0 and line[0] == '#':
                continue
            if '=' in line:
                pair = line.strip().split('#', 1)[0].split('=', 1)
                item = pair[0]
                value = pair[1]
                # noinspection PyBroadException
                try:
                    if item in config:
                        print('Warning: duplicated config item found: {}, updated.'.format((pair[0])))
                    if value[0] == '[' and value[-1] == ']':
                        value_items = list(value[1:-1].split(','))
                        config[item] = value_items
                    else:
                        config[item] = value
                except Exception:
                    print('configuration parsing error, please check correctness of the config file.')
                    exit(1)
        return config

    @staticmethod
    def str2bool(string):
        if string == 'True' or string == 'true' or string == 'TRUE':
            return True
        else:
            return False

    def show_data_summary(self, logger):
        logger.info('++' * 20 + 'CONFIGURATION SUMMARY' + '++' * 20)
        logger.info(' Status:')
        logger.info('     mode                 : {}'.format(self.mode))
        logger.info(' ' + '++' * 20)
        logger.info(' Datasets:')
        logger.info('     datasets         fold: {}'.format(self.datasets_fold))
        logger.info('     train            file: {}'.format(self.train_file))
        logger.info('     validation       file: {}'.format(self.dev_file))
        logger.info('     test             file: {}'.format(self.test_file))
        logger.info('     checkpoints       dir: {}'.format(self.checkpoints_dir))
        logger.info('     log               dir: {}'.format(self.log_dir))
        logger.info(' ' + '++' * 20)
        logger.info('Labeling Scheme:')
        logger.info('     classnames     scheme: {}'.format(self.class_name))
        logger.info('     decision    threshold: {}'.format(self.decision_threshold))
        logger.info(' ' + '++' * 20)
        logger.info(' Training Settings:')
        logger.info('     epoch                : {}'.format(self.epoch))
        logger.info('     batch            size: {}'.format(self.batch_size))
        logger.info('     dropout              : {}'.format(self.dropout))
        logger.info('     learning         rate: {}'.format(self.learning_rate))
        logger.info('     is     early     stop: {}'.format(self.is_early_stop))
        logger.info('     patient              : {}'.format(self.patient))
        logger.info('++' * 20 + 'CONFIGURATION SUMMARY END' + '++' * 20)
        sys.stdout.flush()
