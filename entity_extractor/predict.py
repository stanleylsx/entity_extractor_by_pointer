from tqdm import tqdm
from transformers import BertTokenizer
import torch
import numpy as np


def extract_entities(text, bert_model, model):
    predict_results = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    token_results = tokenizer(text, padding='max_length')
    token_ids = torch.unsqueeze(torch.LongTensor(token_results.get('input_ids')), 0)
    attention_mask = torch.unsqueeze(torch.LongTensor(token_results.get('attention_mask')), 0)
    bert_hidden_states, _ = bert_model(token_ids, attention_mask=attention_mask)
    model_outputs = model(bert_hidden_states)
    for model_output in model_outputs:
        start = np.where(model_output[:, :, 0] > 0.5)
        end = np.where(model_output[:, :, 1] > 0.5)
        for _start, predicate1 in zip(*start):
            for _end, predicate2 in zip(*end):
                if _start <= _end and predicate1 == predicate2:
                    _entity = text[_start: _start + _end + 1]
                    predict_results.append((predicate2, _entity))
                    break
    return list(set(predict_results))


def evaluate(bert_model, model, dev_data):
    """
    评估函数，计算f1、precision、recall
    """
    A, B, C = 1e-10, 1e-10, 1e-10
    T = set()
    for data_row in tqdm(iter(dev_data)):
        R = set(extract_entities(data_row.get('text'), bert_model, model))
        print(R)
        if data_row.get('company') is not None:
            T.add((0, data_row.get('company')))
        if data_row.get('position') is not None:
            T.add((1, data_row.get('position')))
        if data_row.get('detail') is not None:
            T.add((2, data_row.get('detail')))
        A += len(R & T)
        B += len(R)
        C += len(T)
    f1, precision, recall = 2 * A / (B + C), A / B, A / C
    return f1, precision, recall

