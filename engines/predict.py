from tqdm import tqdm
from transformers import BertTokenizer
import torch
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
                    token_list = input_ids[_start: _end + 1]
                    token_list = [token for token in token_list if token != 0]
                    predict_results.setdefault(predicate1, set()).add(str(token_list))
                    break
    return predict_results


def evaluate(configs, bert_model, model, dev_data, device):
    """
    评估函数，分别计算每个类别的f1、precision、recall
    """
    categories = {configs.class_name[index]: index for index in range(0, len(configs.class_name))}
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
                if type(item_text) is str:
                    item_token = tokenizer(item_text).get('input_ids')[1:-1]
                    results.setdefault(class_id, set()).add(str(item_token))
                elif type(item_text) is list:
                    for sub_item in item_text:
                        item_token = tokenizer(sub_item).get('input_ids')[1:-1]
                        results.setdefault(class_id, set()).add(str(item_token))
            else:
                results.setdefault(class_id, set())

        for class_id, token_set in results.items():
            p_token_set = p_results.get(class_id)
            if p_token_set is None:
                # 没预测出来
                p_token_set = set()
            # 预测出来并且正确个数
            counts[class_id]['A'] += len(p_token_set & token_set)
            # 预测出来的结果个数
            counts[class_id]['B'] += len(p_token_set)
            # 真实的结果个数
            counts[class_id]['C'] += len(token_set)
    for class_id, count in counts.items():
        f1, precision, recall = 2 * count['A'] / (count['B'] + count['C']), count['A'] / count['B'], count['A'] / count['C']
        class_name = reverse_categories[class_id]
        results_of_each_entity[class_name]['f1'] = f1
        results_of_each_entity[class_name]['precision'] = precision
        results_of_each_entity[class_name]['recall'] = recall
    return results_of_each_entity
