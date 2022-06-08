# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : configure.py
# @Software: PyCharm

# 模式
# train:训练分类器
# interactive_predict:交互模式
# test:跑测试集
# convert2tf:将torch模型保存为tf框架的pb格式文件
# [train, interactive_predict, test, convert2tf]
mode = 'train'

# 使用GPU设备
use_cuda = True
cuda_device = -1

configure = {
    # 训练数据集
    'train_file': 'data/example_datasets1/train_data.json',
    # 验证数据集
    'dev_file': 'data/example_datasets1/dev_data.json',
    # 没有验证集时，从训练集抽取验证集比例
    # 'validation_rate': 0.15,
    # 测试数据集
    'test_file': '',
    # 使用的模型
    # bp: binary pointer
    # gp: global pointer
    'model_type': 'gp',
    # 模型保存的文件夹
    'checkpoints_dir': 'checkpoints',
    # 类别列表
    'classes': ['company', 'position', 'detail'],
    # decision_threshold
    'decision_threshold': 0.5,
    # 是否使用苏神的多标签分类的损失函数，默认使用BCELoss
    'use_multilabel_categorical_cross_entropy': True,
    # 句子最大长度
    'max_sequence_length': 200,
    # epoch
    'epoch': 50,
    # batch_size
    'batch_size': 16,
    # dropout rate
    'dropout_rate': 0.5,
    # learning_rate
    'learning_rate': 5e-5,
    # 训练是否提前结束微调
    'is_early_stop': True,
    # 训练阶段的patient
    'patient': 5,
}
