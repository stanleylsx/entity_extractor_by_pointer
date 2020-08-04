from transformers import *
import torch
import json

BATCH_SIZE = 64

# Load dataset, tokenizer, model from pretrained model/vocabulary
# sentence = '深圳市瑞致达科技有限公司\n设备管理专员 2006年11月-2019年12月\n内容：1、与一流的开发团队协同工作，根据需求对游戏进行功能、' \
#            '性能、兼容、接口测试并且反馈游戏可玩性、易玩性；2、能独立执行项目测试，设计测试方案、测试用例等相关文档，能够准确定位测试重点；' \
#            '3、协助客户端、server技术人员定位并解决复杂的技术问题；4、根据测试结果编写测试报告并向上级汇报。'
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# model = BertModel.from_pretrained('bert-base-chinese')
# input_ids = torch.tensor([tokenizer.encode(sentence)])
# all_hidden_states, all_attentions = model(input_ids)[-2:]
# print(all_hidden_states, all_hidden_states.shape)


class DataGenerator:
    def __init__(self, data, batch_size=BATCH_SIZE):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def prepare_data(self):
        print(self.data)


if __name__ == '__main__':
    train_data = json.load(open('data/train_data.json'))
    dev_data = json.load(open('data/dev_data.json'))
    data_generator = DataGenerator(train_data)
    data_generator.prepare_data()