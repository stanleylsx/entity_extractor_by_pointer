import json
import math
import re
import codecs


def split_data(all_list, ratio=0.9):
    """
    分割训练集与验证集
    """
    num = len(all_list)
    offset = int(num * ratio)
    if num == 0 or offset < 1:
        return [], all_list
    train = all_list[:offset]
    test = all_list[offset:]
    return train, test


def split_text(text):
    """
    根据句末符号最大程度分割文本
    """
    index_list = [it.span()[0] for it in re.finditer(r'[。;；\s]', text)]
    index_list.insert(0, 0)
    text_segment = []
    split_count = 500
    split_indices = []
    start_p, end_p = 0, 0
    while end_p < len(index_list):
        if index_list[end_p] - index_list[start_p] > split_count:
            start_p = end_p - 1
            split_indices.append(index_list[start_p] + 1)
        end_p = end_p + 1
    split_indices.insert(0, 0)
    split_between = [split_indices[i: i+2] for i in range(len(split_indices)-1)]
    for it in split_between:
        text_segment.append(text[it[0]: it[1]])
    text_segment.append(text[split_between[-1][1]:])
    return text_segment


def transfer_row_data():
    with open('resumes_groundtruth.json', encoding='utf-8') as resume_text:
        resume_data = json.loads(resume_text.read())
    work_experience_list_2 = []
    for key, value in resume_data.items():
        work_experience_list = value.get('工作经历')
        for exp in work_experience_list:
            work_experience_list_2.append(exp)
    n = math.ceil(len(work_experience_list_2)/4)
    work_experience_split = [work_experience_list_2[i:i + n] for i in range(0, len(work_experience_list_2), n)]
    competition_data_list = []
    for sub_list in work_experience_split:
        index = work_experience_split.index(sub_list)
        if index == 0:
            # 公司+时间+职位+描述
            for item in sub_list:
                text = item['工作单位'] + ' ' + item['工作时间'] + ' ' + item['职务'] + ' ' + item['工作内容']
                if len(text) <= 510:
                    company, date, position, detail = item['工作单位'], item['工作时间'], item['职务'], item['工作内容']
                    competition_data_list.append({'text': text, 'company': company, 'date': date, 'position': position, 'detail': detail})
                else:
                    detail = item['工作内容'][:20]
                    text_segments = split_text(text)
                    start = re.search(detail, text_segments[0]).span()[0]
                    company, date, position, detail = item['工作单位'], item['工作时间'], item['职务'], text_segments[0][start:]
                    competition_data_list.append({'text': text_segments[0], 'company': company, 'date': date, 'position': position, 'detail': detail})
                    for text_rest in text_segments[1:]:
                        competition_data_list.append({'text': text_rest, 'detail': text_rest})
        elif index == 1:
            # 时间+公司+职位+描述
            for item in sub_list:
                if '职务' in item:
                    text = item['工作时间'] + ' ' + item['工作单位'] + ' ' + item['职务'] + ' ' + item['工作内容']
                    if len(text) <= 510:
                        company, date, position, detail = item['工作单位'], item['工作时间'], item['职务'], item['工作内容']
                        competition_data_list.append({'text': text, 'company': company, 'date': date, 'position': position, 'detail': detail})
                    else:
                        detail = item['工作内容'][:20]
                        text_segments = split_text(text)
                        start = re.search(detail, text_segments[0]).span()[0]
                        company, date, position, detail = item['工作单位'], item['工作时间'], item['职务'], text_segments[0][
                                                                                                  start:]
                        competition_data_list.append(
                            {'text': text_segments[0], 'company': company, 'date': date, 'position': position,
                             'detail': detail})
                        for text_rest in text_segments[1:]:
                            competition_data_list.append({'text': text_rest, 'detail': text_rest})
                else:
                    text = item['工作时间'] + ' ' + item['工作单位'] + ' ' + item['工作内容']
                    if len(text) <= 510:
                        company, date, detail = item['工作单位'], item['工作时间'], item['工作内容']
                        competition_data_list.append({'text': text, 'company': company, 'date': date, 'detail': detail})
                    else:
                        detail = item['工作内容'][:20]
                        text_segments = split_text(text)
                        start = re.search(detail, text_segments[0]).span()[0]
                        company, date, detail = item['工作单位'], item['工作时间'], text_segments[0][start:]
                        competition_data_list.append({'text': text_segments[0], 'company': company, 'date': date, 'position': position, 'detail': detail})
                        for text_rest in text_segments[1:]:
                            competition_data_list.append({'text': text_rest, 'detail': text_rest})

        elif index == 2:
            # 公司+职位+时间+描述
            for item in sub_list:
                text = item['工作单位'] + ' ' + item['职务'] + ' ' + item['工作时间'] + ' ' + item['工作内容']
                if len(text) <= 510:
                    company, date, position, detail = item['工作单位'], item['工作时间'], item['职务'], item['工作内容']
                    competition_data_list.append({'text': text, 'company': company, 'date': date, 'position': position, 'detail': detail})
                else:
                    detail = item['工作内容'][:20]
                    text_segments = split_text(text)
                    start = re.search(detail, text_segments[0]).span()[0]
                    company, date, position, detail = item['工作单位'], item['工作时间'], item['职务'], text_segments[0][start:]
                    competition_data_list.append({'text': text_segments[0], 'company': company, 'date': date, 'position': position, 'detail': detail})
                    for text_rest in text_segments[1:]:
                        competition_data_list.append({'text': text_rest, 'detail': text_rest})
        elif index == 3:
            # 时间+职位+公司+描述
            for item in sub_list:
                text = item['工作时间'] + ' ' + item['职务'] + ' ' + item['工作单位'] + ' ' + item['工作内容']
                if len(text) <= 510:
                    company, date, position, detail = item['工作单位'], item['工作时间'], item['职务'], item['工作内容']
                    competition_data_list.append({'text': text, 'company': company, 'date': date, 'position': position, 'detail': detail})
                else:
                    detail = item['工作内容'][:20]
                    text_segments = split_text(text)
                    start = re.search(detail, text_segments[0]).span()[0]
                    company, date, position, detail = item['工作单位'], item['工作时间'], item['职务'], text_segments[0][start:]
                    competition_data_list.append({'text': text_segments[0], 'company': company, 'date': date, 'position': position, 'detail': detail})
                    for text_rest in text_segments[1:]:
                        competition_data_list.append({'text': text_rest, 'detail': text_rest})

    competition_data_list, dev_data_list = split_data(competition_data_list)

    with codecs.open('train_data.json', 'w', encoding='utf-8') as f:
        json.dump(competition_data_list, f, indent=4, ensure_ascii=False)

    with codecs.open('dev_data.json', 'w', encoding='utf-8') as f:
        json.dump(dev_data_list, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    transfer_row_data()
