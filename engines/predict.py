# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: PyCharm
import torch
import os
import time
import json
from torch.utils.data import DataLoader
from tqdm import tqdm


class Predictor:
    def __init__(self, configs, data_manager, device, logger):
        self.device = device
        self.configs = configs
        self.data_manager = data_manager
        self.logger = logger
        self.checkpoints_dir = configs['checkpoints_dir']
        self.model_name = configs['model_name']
        num_labels = len(self.data_manager.categories)
        if configs['model_type'] == 'bp':
            from engines.models.BinaryPointer import BinaryPointer
            self.model = BinaryPointer(num_labels=num_labels).to(device)
        else:
            from engines.models.GlobalPointer import EffiGlobalPointer
            self.model = EffiGlobalPointer(num_labels=num_labels, device=device).to(device)
        self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, self.model_name)))
        self.model.eval()

    def predict_one(self, sentence):
        """
        预测接口
        """
        start_time = time.time()
        encode_results = self.data_manager.tokenizer(sentence, padding='max_length')
        input_ids = encode_results.get('input_ids')
        token_ids = torch.unsqueeze(torch.LongTensor(input_ids), 0).to(self.device)
        attention_mask = torch.unsqueeze(torch.LongTensor(encode_results.get('attention_mask')), 0).to(self.device)
        segment_ids = torch.unsqueeze(torch.LongTensor(encode_results.get('token_type_ids')), 0).to(self.device)
        logits, _ = self.model(token_ids, attention_mask, segment_ids)
        logit = torch.squeeze(logits.to('cpu'))
        results = self.data_manager.extract_entities(sentence, logit)
        self.logger.info('predict time consumption: %.3f(ms)' % ((time.time() - start_time) * 1000))
        results_dict = {}
        for class_id, result_set in results.items():
            results_dict[self.data_manager.reverse_categories[class_id]] = list(result_set)
        return results_dict

    def predict_test(self):
        test_file = self.configs['test_file']
        if test_file == '' or not os.path.exists(test_file):
            self.logger.info('test dataset does not exist!')
            return
        test_data = json.load(open(test_file, encoding='utf-8'))
        test_loader = DataLoader(
            dataset=test_data,
            batch_size=self.data_manager.batch_size,
            collate_fn=self.data_manager.prepare_data,
        )
        counts = {}
        results_of_each_entity = {}
        for class_name, class_id in self.data_manager.categories.items():
            counts[class_id] = {'A': 0.0, 'B': 1e-10, 'C': 1e-10}
            class_name = self.data_manager.reverse_categories[class_id]
            results_of_each_entity[class_name] = {}

        with torch.no_grad():
            self.model.eval()
            self.logger.info('start evaluate engines...')
            for batch in tqdm(test_loader):
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

    def convert_torch_to_tf(self):
            import onnx
            from onnx_tf.backend import prepare
            max_sequence_length = self.data_manager.max_sequence_length
            dummy_input = torch.ones([1, max_sequence_length]).to('cpu').long()
            dummy_input = (dummy_input, dummy_input, dummy_input)
            onnx_path = self.checkpoints_dir + '/model.onnx'
            torch.onnx.export(self.model.to('cpu'), dummy_input, f=onnx_path, opset_version=13,
                              input_names=['tokens', 'attentions', 'types'], output_names=['logits', 'probs'],
                              do_constant_folding=False,
                              dynamic_axes={'tokens': {0: 'batch_size'}, 'attentions': {0: 'batch_size'},
                                            'types': {0: 'batch_size'}, 'logits': {0: 'batch_size'},
                                            'probs': {0: 'batch_size'}})
            model_onnx = onnx.load(onnx_path)
            tf_rep = prepare(model_onnx)
            tf_rep.export_graph(self.checkpoints_dir + '/model.pb')
            self.logger.info('convert torch to tensorflow pb successful...')
