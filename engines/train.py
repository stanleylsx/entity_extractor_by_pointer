# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py
# @Software: PyCharm
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import torch
import time
import os


class Train:
    def __init__(self, configs, data_manager, device, logger):
        self.configs = configs
        self.device = device
        self.logger = logger
        self.data_manager = data_manager
        self.batch_size = self.configs['batch_size']
        self.num_labels = len(configs['classes'])
        self.checkpoints_dir = configs['checkpoints_dir']
        self.model_name = configs['model_name']
        self.epoch = configs['epoch']

        learning_rate = configs['learning_rate']

        if configs['model_type'] == 'bp':
            from engines.models.BinaryPointer import BinaryPointer
            self.model = BinaryPointer(num_labels=self.num_labels).to(device)
        else:
            from engines.models.GlobalPointer import EffiGlobalPointer
            self.model = EffiGlobalPointer(num_labels=self.num_labels, device=device).to(device)

        if configs['use_gan']:
            if configs['gan_method'] == 'fgm':
                from engines.utils.gan_utils import FGM
                self.gan = FGM(self.model)
            else:
                from engines.utils.gan_utils import PGD
                self.gan = PGD(self.model)

        params = list(self.model.parameters())
        optimizer_type = configs['optimizer']
        if optimizer_type == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(params, lr=learning_rate)
        elif optimizer_type == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(params, lr=learning_rate)
        elif optimizer_type == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(params, lr=learning_rate)
        elif optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(params, lr=learning_rate)
        elif optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(params, lr=learning_rate)
        elif optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(params, lr=learning_rate)
        else:
            raise Exception('optimizer_type does not exist')

        if configs['use_multilabel_categorical_cross_entropy']:
            from engines.utils.losses import MultilabelCategoricalCrossEntropy
            self.loss_function = MultilabelCategoricalCrossEntropy()
        else:
            self.loss_function = torch.nn.BCEWithLogitsLoss(reduction='none')

        if os.path.exists(os.path.join(self.checkpoints_dir, self.model_name)):
            logger.info('Resuming from checkpoint...')
            self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, self.model_name)))
            optimizer_checkpoint = torch.load(os.path.join(self.checkpoints_dir, self.model_name + '.optimizer'))
            self.optimizer.load_state_dict(optimizer_checkpoint['optimizer'])
        else:
            logger.info('Initializing from scratch.')

    def calculate_loss(self, logits, labels, attention_mask):
        batch_size = logits.size(0)
        if self.configs['use_multilabel_categorical_cross_entropy']:
            if self.configs['model_type'] == 'bp':
                num_labels = self.num_labels * 2
            else:
                num_labels = self.num_labels
            model_output = logits.reshape(batch_size * num_labels, -1)
            label_vectors = labels.reshape(batch_size * num_labels, -1)
            loss = self.loss_function(model_output, label_vectors)
        else:
            if self.configs['model_type'] == 'bp':
                loss = self.loss_function(logits, labels)
                loss = torch.sum(torch.mean(loss, 3), 2)
                loss = torch.sum(loss * attention_mask) / torch.sum(attention_mask)
            else:
                model_output = logits.reshape(batch_size * self.num_labels, -1)
                label_vectors = labels.reshape(batch_size * self.num_labels, -1)
                loss = self.loss_function(model_output, label_vectors).mean()
        return loss

    def train(self):
        train_file = self.configs['train_file']
        dev_file = self.configs['dev_file']
        train_data = json.load(open(train_file, encoding='utf-8'))

        if dev_file == '':
            self.logger.info('generate validation dataset...')
            validation_rate = self.configs['validation_rate']
            ratio = 1 - validation_rate
            train_data, dev_data = train_data[:int(ratio * len(train_data))], train_data[int(ratio * len(train_data)):]
        else:
            dev_data = json.load(open(dev_file, encoding='utf-8'))

        self.logger.info('loading train data...')
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            collate_fn=self.data_manager.prepare_data,
            shuffle=True
        )
        dev_loader = DataLoader(
            dataset=dev_data,
            batch_size=self.batch_size,
            collate_fn=self.data_manager.prepare_data,
        )

        best_f1 = 0
        best_epoch = 0
        unprocessed = 0
        step_total = self.epoch * len(train_loader)
        global_step = 0
        scheduler = None

        if self.configs['warmup']:
            scheduler_type = self.configs['scheduler_type']
            if self.configs['num_warmup_steps'] == -1:
                num_warmup_steps = step_total * 0.1
            else:
                num_warmup_steps = self.configs['num_warmup_steps']

            if scheduler_type == 'linear':
                from transformers.optimization import get_linear_schedule_with_warmup
                scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=step_total)
            elif scheduler_type == 'cosine':
                from transformers.optimization import get_cosine_schedule_with_warmup
                scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=step_total)
            else:
                raise Exception('scheduler_type does not exist')

        very_start_time = time.time()
        for i in range(self.epoch):
            self.logger.info('\nepoch:{}/{}'.format(i + 1, self.epoch))
            self.model.train()
            start_time = time.time()
            step, loss, loss_sum = 0, 0.0, 0.0
            for batch in tqdm(train_loader):
                _, _, token_ids, token_type_ids, attention_mask, label_vectors = batch
                token_ids = token_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                label_vectors = label_vectors.to(self.device)
                self.optimizer.zero_grad()
                logits, _ = self.model(token_ids, attention_mask, token_type_ids)
                loss = self.calculate_loss(logits, label_vectors, attention_mask)
                loss.backward()
                loss_sum += loss.item()
                if self.configs['use_gan']:
                    k = self.configs['attack_round']
                    if self.configs['gan_method'] == 'fgm':
                        self.gan.attack()
                        logits, _ = self.model(token_ids, attention_mask, token_type_ids)
                        loss = self.calculate_loss(logits, label_vectors, attention_mask)
                        loss.backward()
                        self.gan.restore()  # 恢复embedding参数
                    else:
                        self.gan.backup_grad()
                        for t in range(k):
                            self.gan.attack(is_first_attack=(t == 0))
                            if t != k - 1:
                                self.model.zero_grad()
                            else:
                                self.gan.restore_grad()
                            logits, _ = self.model(token_ids, attention_mask, token_type_ids)
                            loss = self.calculate_loss(logits, label_vectors, attention_mask)
                            loss.backward()
                        self.gan.restore()
                self.optimizer.step()

                if self.configs['warmup']:
                    scheduler.step()

                if step % self.configs['print_per_batch'] == 0 and step != 0:
                    avg_loss = loss_sum / (step + 1)
                    self.logger.info('training_loss:%f' % avg_loss)

                step = step + 1
                global_step = global_step + step

            f1 = self.validate(dev_loader)
            time_span = (time.time() - start_time) / 60
            self.logger.info('time consumption:%.2f(min)' % time_span)
            if f1 >= best_f1:
                unprocessed = 0
                best_f1 = f1
                best_epoch = i + 1
                optimizer_checkpoint = {'optimizer': self.optimizer.state_dict()}
                torch.save(optimizer_checkpoint, os.path.join(self.checkpoints_dir, self.model_name + '.optimizer'))
                torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, self.model_name))
                self.logger.info('saved model successful...')
            else:
                unprocessed += 1
            aver_loss = loss_sum / step
            self.logger.info(
                'aver_loss: %.4f, f1: %.4f, best_f1: %.4f, best_epoch: %d \n' % (aver_loss, f1, best_f1, best_epoch))
            if self.configs['is_early_stop']:
                if unprocessed > self.configs['patient']:
                    self.logger.info('early stopped, no progress obtained within {} epochs'.format(
                        self.configs['patient']))
                    self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1, best_epoch))
                    self.logger.info('total training time consumption: %.3f(min)' % (
                            (time.time() - very_start_time) / 60))
                    return
        self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1, best_epoch))
        self.logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))

    def validate(self, dev_loader):
        counts = {}
        results_of_each_entity = {}
        for class_name, class_id in self.data_manager.categories.items():
            counts[class_id] = {'A': 0.0, 'B': 1e-10, 'C': 1e-10}
            class_name = self.data_manager.reverse_categories[class_id]
            results_of_each_entity[class_name] = {}

        with torch.no_grad():
            self.model.eval()
            self.logger.info('start evaluate engines...')
            for batch in tqdm(dev_loader):
                texts, entity_results, token_ids, segment_ids, attention_mask, _ = batch
                token_ids = token_ids.to(self.device)
                segment_ids = segment_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                logits, _ = self.model(token_ids, attention_mask, segment_ids)
                logits = logits.to('cpu')
                for text, logit, entity_result in zip(texts, logits, entity_results):
                    p_results = self.data_manager.extract_entities(text, logit)
                    for class_id, entity_set in entity_result.items():
                        p_entity_set = p_results.get(class_id)
                        if p_entity_set is None:
                            # 没预测出来
                            p_entity_set = set()
                        # 预测出来并且正确个数
                        counts[class_id]['A'] += len(p_entity_set & entity_set)
                        # 预测出来的结果个数
                        counts[class_id]['B'] += len(p_entity_set)
                        # 真实的结果个数
                        counts[class_id]['C'] += len(entity_set)
        for class_id, count in counts.items():
            f1, precision, recall = 2 * count['A'] / (
                    count['B'] + count['C']), count['A'] / count['B'], count['A'] / count['C']
            class_name = self.data_manager.reverse_categories[class_id]
            results_of_each_entity[class_name]['f1'] = f1
            results_of_each_entity[class_name]['precision'] = precision
            results_of_each_entity[class_name]['recall'] = recall

        f1 = 0.0
        for class_id, performance in results_of_each_entity.items():
            f1 += performance['f1']
            # 打印每个类别的指标
            self.logger.info('class_name: %s, precision: %.4f, recall: %.4f, f1: %.4f'
                        % (class_id, performance['precision'], performance['recall'], performance['f1']))
        # 这里算得是所有类别的平均f1值
        f1 = f1 / len(results_of_each_entity)
        return f1
