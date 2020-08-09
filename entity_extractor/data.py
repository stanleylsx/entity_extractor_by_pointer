from transformers import *
from torch.utils.data import Dataset
import torch
import numpy as np
import re

BATCH_SIZE = 64
EPOCH_NUM = 10


class DataGenerator:
    def __init__(self, data, batch_size=BATCH_SIZE):
        self.data = data
        self.batch_size = batch_size
        self.categories = {'company': 1, 'position': 2, 'detail': 3}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    @staticmethod
    def get_index(text_token, token):
        text_token_str = '#'.join([str(index) for index in text_token])
        token_str = '#'.join([str(index) for index in token])
        start_str_index = re.search(token_str, text_token_str).span()[0]
        start_index = text_token_str[:start_str_index].count('#')
        end_index = start_index + len(token)
        return start_index, end_index

    def prepare_data(self):
        sentence_vectors = []
        segment_vectors = []
        attention_mask_vectors = []
        start_vectors = []
        end_vectors = []
        for item in self.data:
            text = item.get('text')
            company = item.get('company')
            position = item.get('position')
            detail = item.get('detail')
            token_results = self.tokenizer(text, padding='max_length')
            token_ids = token_results.get('input_ids')
            segment_ids = token_results.get('token_type_ids')
            attention_mask = token_results.get('attention_mask')
            start_vector, end_vector = [0] * 512, [0] * 512
            if company is not None:
                company_ids = self.tokenizer(company).get('input_ids')
                company_ids = company_ids[1:-1]
                start_index, end_index = self.get_index(token_ids, company_ids)
                start_vector[start_index] = self.categories.get('company')
                end_vector[end_index - 1] = self.categories.get('company')
            if position is not None:
                position_ids = self.tokenizer(position).get('input_ids')
                position_ids = position_ids[1:-1]
                start_index, end_index = self.get_index(token_ids, position_ids)
                start_vector[start_index] = self.categories.get('position')
                end_vector[end_index - 1] = self.categories.get('position')
            if detail is not None:
                detail_ids = self.tokenizer(detail).get('input_ids')
                detail_ids = detail_ids[1:-1]
                start_index, end_index = self.get_index(token_ids, detail_ids)
                start_vector[start_index] = self.categories.get('detail')
                end_vector[end_index - 1] = self.categories.get('detail')
            sentence_vectors.append(token_ids)
            segment_vectors.append(segment_ids)
            attention_mask_vectors.append(attention_mask)
            start_vectors.append(start_vector)
            end_vectors.append(end_vector)
            # print(text)
            # for item in zip(input_ids, start_vector, end_vector):
            #     print(item)
        sentence_vectors_np = np.array(sentence_vectors)
        segment_vectors_np = np.array(segment_vectors)
        attention_mask_vectors_np = np.array(attention_mask_vectors)
        start_vectors_np = np.array(start_vectors)
        end_vectors_np = np.array(end_vectors)
        return [sentence_vectors_np, segment_vectors_np, attention_mask_vectors_np, start_vectors_np, end_vectors_np]


class MyDataset(Dataset):
    """
    下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, _sentence_vectors, _segment_vectors, _attention_mask_vectors, _start_vectors, _end_vectors):
        self.sentence_vectors = _sentence_vectors
        self.segment_vectors = _segment_vectors
        self.attention_mask_vectors = _attention_mask_vectors
        self.start_vectors = _start_vectors
        self.end_vectors = _end_vectors
        self.len = len(self.sentence_vectors)

    def __getitem__(self, index):
        return self.sentence_vectors[index], self.segment_vectors[index], self.attention_mask_vectors[index],\
               self.start_vectors[index], self.end_vectors[index]

    def __len__(self):
        return self.len


def collate_fn(data):
    """
    如何取样本的，定义自己的函数来准确地实现想要的功能
    """
    sentence = np.array([item[0] for item in data], np.int32)
    segment = np.array([item[1] for item in data], np.int32)
    attention_mask = np.array([item[2] for item in data], np.int32)
    start_vec = np.array([item[3] for item in data], np.int32)
    end_vec = np.array([item[4] for item in data], np.int32)
    return {'sentence': torch.LongTensor(sentence),
            'segment': torch.LongTensor(segment),
            'attention_mask': torch.LongTensor(attention_mask),
            'start_vec': torch.LongTensor(start_vec),
            'end_vec': torch.LongTensor(end_vec)}



