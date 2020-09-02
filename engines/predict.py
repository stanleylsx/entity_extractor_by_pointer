from tqdm import tqdm
from transformers import BertTokenizer
from utils.rematch import rematch
import torch
import numpy as np


def extract_entities(configs, tokenizer, text, bert_model, model, device):
    """
    从验证集中预测到相关实体
    """
    predict_results = {}
    encode_results = tokenizer(text, padding='max_length')
    input_ids = encode_results.get('input_ids')
    token = tokenizer.convert_ids_to_tokens(input_ids)
    mapping = rematch(text, token)
    token_ids = torch.unsqueeze(torch.LongTensor(input_ids), 0).to(device)
    attention_mask = torch.unsqueeze(torch.LongTensor(encode_results.get('attention_mask')), 0).to(device)
    bert_hidden_states = bert_model(token_ids, attention_mask=attention_mask)[0].to(device)
    model_outputs = model(bert_hidden_states).detach().to('cpu')
    decision_threshold = float(configs.decision_threshold)
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
        p_results = extract_entities(configs, tokenizer, data_row.get('text'), bert_model, model, device)
        for class_name, class_id in categories.items():
            item_text = data_row.get(class_name)
            if item_text is not None:
                if type(item_text) is str:
                    results.setdefault(class_id, set()).add(item_text)
                elif type(item_text) is list:
                    for sub_item in item_text:
                        results.setdefault(class_id, set()).add(sub_item)
            else:
                results.setdefault(class_id, set())

        for class_id, entity_set in results.items():
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
        f1, precision, recall = 2 * count['A'] / (count['B'] + count['C']), count['A'] / count['B'], count['A'] / count['C']
        class_name = reverse_categories[class_id]
        results_of_each_entity[class_name]['f1'] = f1
        results_of_each_entity[class_name]['precision'] = precision
        results_of_each_entity[class_name]['recall'] = recall
    return results_of_each_entity


def predict_one(configs, tokenizer, sentence, bert_model, model, device):
    """
    预测接口
    """
    categories = {configs.class_name[index]: index for index in range(0, len(configs.class_name))}
    reverse_categories = {class_id: class_name for class_name, class_id in categories.items()}
    results = extract_entities(configs, tokenizer, sentence, bert_model, model, device)
    results_dict = {}
    for class_id, result_set in results.items():
        results_dict[reverse_categories[class_id]] = list(result_set)
    return results_dict


