import re


def judge_split(index_list, split_count):
    for index in range(0, len(index_list) - 1):
        if index_list[index + 1] - index_list[index] > split_count:
            return True
    return False


def split_text(text):
    """
    根据句末符号最大程度分割文本
    """
    split_count = 500
    index_list = [it.start() for it in re.finditer(r'[。;；!！\s]', text)]
    index_list.insert(0, 0)
    index_list.append(len(text))

    if judge_split(index_list, split_count):
        """分不开就直接按字切分"""
        text_segment = re.findall(r'.{' + str(split_count) + '}', text)
        text_segment.append(text[(len(text_segment) * split_count):])
        return text_segment
    text_segment = []
    split_indices = []
    start_p, end_p = 0, 0
    while end_p < len(index_list):
        if index_list[end_p] - index_list[start_p] > split_count:
            start_p = end_p - 1
            split_indices.append(index_list[start_p] + 1)
        end_p = end_p + 1
    split_indices.append(index_list[-1])
    split_indices.insert(0, 0)
    split_between = [split_indices[i: i+2] for i in range(len(split_indices)-1)]
    for it in split_between:
        text_segment.append(text[it[0]: it[1]])
    return text_segment
