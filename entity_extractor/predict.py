from tqdm import tqdm
from transformers import BertTokenizer
import torch
import re
import numpy as np


def extract_entities(tokenizer, text, bert_model, model, device):
    """
    从验证集中预测到相关实体
    """
    predict_results = {}
    token_results = tokenizer(text, padding='max_length')
    input_ids = token_results.get('input_ids')
    token_ids = torch.unsqueeze(torch.LongTensor(input_ids), 0).to(device)
    attention_mask = torch.unsqueeze(torch.LongTensor(token_results.get('attention_mask')), 0).to(device)
    bert_hidden_states = bert_model(token_ids, attention_mask=attention_mask)[0].to(device)
    model_outputs = model(bert_hidden_states).detach().to('cpu')
    for model_output in model_outputs:
        start = np.where(model_output[:, :, 0] > 0.5)
        end = np.where(model_output[:, :, 1] > 0.5)
        for _start, predicate1 in zip(*start):
            for _end, predicate2 in zip(*end):
                if _start <= _end and predicate1 == predicate2:
                    _entity = tokenizer.decode(input_ids[_start: _end + 1])
                    predict_results.setdefault(predicate1, set()).add(re.sub(r'\s', '', _entity))
                    break
    return predict_results


def evaluate(bert_model, model, dev_data, device):
    """
    评估函数，分别计算每个类别的f1、precision、recall
    """
    categories = {'company': 0, 'position': 1, 'detail': 2}
    reverse_categories = {class_id: class_name for class_name, class_id in categories.items()}
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    counts = {}
    results_of_each_entity = {}
    for class_name, class_id in categories.items():
        counts[class_id] = {'A': 0.0, 'B': 1e-10, 'C': 1e-10}
        class_name = reverse_categories[class_id]
        results_of_each_entity[class_name] = {}

    for data_row in tqdm(iter(dev_data)):
        results = {}
        p_results = extract_entities(tokenizer, data_row.get('text'), bert_model, model, device)
        for class_name, class_id in categories.items():
            item_text = data_row.get(class_name)
            if item_text is not None:
                results.setdefault(class_id, set()).add(re.sub(r'\s', '', item_text))
            else:
                results.setdefault(class_id, set())

        for class_id, text_set in results.items():
            p_text_set = p_results.get(class_id)
            if p_text_set is None:
                # 没预测出来
                p_text_set = set()
            # 预测出来并且正确个数
            counts[class_id]['A'] += len(p_text_set & text_set)
            # 预测出来的结果个数
            counts[class_id]['B'] += len(p_text_set)
            # 真实的结果个数
            counts[class_id]['C'] += len(text_set)
    for class_id, count in counts.items():
        f1, precision, recall = 2 * count['A'] / (count['B'] + count['C']), count['A'] / count['B'], count['A'] / count['C']
        class_name = reverse_categories[class_id]
        results_of_each_entity[class_name]['f1'] = f1
        results_of_each_entity[class_name]['precision'] = precision
        results_of_each_entity[class_name]['recall'] = recall
    return results_of_each_entity
