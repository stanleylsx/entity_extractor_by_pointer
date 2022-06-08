# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: PyCharm
import torch
import os


class Predictor:
    def __init__(self, configs, data_manager, device, logger):
        self.device = device
        self.data_manager = data_manager
        self.logger = logger
        num_labels = len(self.data_manager.categories)
        from engines.models.BinaryPointer import BinaryPointer
        self.model = BinaryPointer(num_labels=num_labels).to(device)
        self.model.load_state_dict(torch.load(os.path.join(configs['checkpoints_dir'], 'best_model.pkl')))
        self.model.eval()

    def predict_one(self, sentence):
        """
        预测接口
        """
        encode_results = self.data_manager.tokenizer(sentence, padding='max_length')
        input_ids = encode_results.get('input_ids')
        token_ids = torch.unsqueeze(torch.LongTensor(input_ids), 0).to(self.device)
        attention_mask = torch.unsqueeze(torch.LongTensor(encode_results.get('attention_mask')), 0).to(self.device)
        segment_ids = torch.unsqueeze(torch.LongTensor(encode_results.get('token_type_ids')), 0).to(self.device)
        model_outputs = self.model(token_ids, attention_mask, segment_ids).detach().to('cpu')
        model_output = torch.squeeze(model_outputs)
        results = self.data_manager.extract_entities(sentence, model_output)
        results_dict = {}
        for class_id, result_set in results.items():
            results_dict[self.data_manager.reverse_categories[class_id]] = list(result_set)
        return results_dict
