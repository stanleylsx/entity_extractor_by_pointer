# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py
# @Software: PyCharm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from tqdm import tqdm
from engines.utils.rematch import rematch
import torch
import numpy as np
import re


class DataManager:
    def __init__(self, configs, logger):
        self.logger = logger
        self.configs = configs
        self.batch_size = configs['batch_size']
        self.max_sequence_length = configs['max_sequence_length']
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        self.classes = configs['classes']
        self.num_labels = len(self.classes)
        self.categories = {configs['classes'][index]: index for index in range(0, len(configs['classes']))}
        self.reverse_categories = {class_id: class_name for class_name, class_id in self.categories.items()}

    @staticmethod
    def get_index(text_token, token):
        text_token_str = '#'.join([str(index) for index in text_token])
        token_str = '#'.join([str(index) for index in token])
        start_str_index = re.search(token_str, text_token_str).start()
        start_index = text_token_str[:start_str_index].count('#')
        end_index = start_index + len(token)
        return start_index, end_index - 1

    def padding(self, token):
        if len(token) < self.max_sequence_length:
            token += [0 for _ in range(self.max_sequence_length - len(token))]
        else:
            token = token[:self.max_sequence_length]
        return token

    def prepare_data(self, data):
        sentence_vectors = []
        segment_vectors = []
        attention_mask_vectors = []
        entity_vectors = []
        for item in tqdm(data):
            text = item.get('text')
            token_results = self.tokenizer(text)
            token_ids = self.padding(token_results.get('input_ids'))
            segment_ids = self.padding(token_results.get('token_type_ids'))
            attention_mask = self.padding(token_results.get('attention_mask'))
            entity_vector = np.zeros((len(token_ids), len(self.categories), 2))
            try:
                for class_name, class_id in self.categories.items():
                    item_text = item.get(class_name)
                    if item_text is not None:
                        if type(item_text) is str:
                            item_token = self.tokenizer(item_text).get('input_ids')
                            item_token = item_token[1:-1]
                            start_index, end_index = self.get_index(token_ids, item_token)
                            entity_vector[start_index, class_id, 0] = 1
                            entity_vector[end_index, class_id, 1] = 1
                        elif type(item_text) is list:
                            for sub_item in item_text:
                                item_token = self.tokenizer(sub_item).get('input_ids')
                                item_token = item_token[1:-1]
                                start_index, end_index = self.get_index(token_ids, item_token)
                                entity_vector[start_index, class_id, 0] = 1
                                entity_vector[end_index, class_id, 1] = 1
            except AttributeError:
                continue
            else:
                sentence_vectors.append(token_ids)
                segment_vectors.append(segment_ids)
                attention_mask_vectors.append(attention_mask)
                entity_vectors.append(entity_vector)
        sentence_vectors = torch.tensor(sentence_vectors)
        segment_vectors = torch.tensor(segment_vectors)
        attention_mask_vectors = torch.tensor(attention_mask_vectors)
        entity_vectors = torch.tensor(entity_vectors)
        dataset = TensorDataset(sentence_vectors, segment_vectors, attention_mask_vectors, entity_vectors)
        return dataset

    def extract_entities(self, text, model_outputs):
        """
        从验证集中预测到相关实体
        """
        predict_results = {}
        encode_results = self.tokenizer(text, padding='max_length')
        input_ids = encode_results.get('input_ids')
        token = self.tokenizer.convert_ids_to_tokens(input_ids)
        mapping = rematch(text, token)
        decision_threshold = float(self.configs['decision_threshold'])
        for model_output in model_outputs:
            start = np.where(model_output[:, :, 0] > decision_threshold)
            end = np.where(model_output[:, :, 1] > decision_threshold)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        if len(mapping[_start]) > 0 and len(mapping[_end]) > 0:
                            start_in_text = mapping[_start][0]
                            end_in_text = mapping[_end][-1]
                            entity_text = text[start_in_text: end_in_text + 1]
                            predict_results.setdefault(predicate1, set()).add(entity_text)
                        break
        return predict_results
