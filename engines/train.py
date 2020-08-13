from engines.model import Model
from transformers import AdamW, BertModel
from tqdm import tqdm
from torch.utils.data import DataLoader
from engines.data import DataGenerator, MyDataset, collate_fn
from engines.predict import evaluate
import json
import torch
import os


def train(configs, logger):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_file = configs.datasets_fold + '/' + configs.train_file
    dev_file = configs.datasets_fold + '/' + configs.dev_file
    train_data = json.load(open(train_file, encoding='utf-8'))
    dev_data = json.load(open(dev_file, encoding='utf-8'))
    train_data_generator = DataGenerator(configs, train_data, logger=logger)
    logger.info('dev_data_length:{}\n'.format(len(dev_data)))
    sentence, segment, attention_mask, entity_vectors = train_data_generator.prepare_data()
    torch_dataset = MyDataset(sentence, segment, attention_mask, entity_vectors)
    loader = DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=configs.batch_size,  # mini batch size
        shuffle=True,  # random shuffle for training
        num_workers=0,
        collate_fn=collate_fn,  # subprocesses for loading data
    )
    learning_rate = configs.learning_rate
    adam_epsilon = 1e-05
    num_labels = len(configs.class_name)
    model = Model(hidden_size=768, num_labels=num_labels).to(device)
    bert_model = BertModel.from_pretrained('bert-base-chinese').to(device)
    params = list(model.parameters())
    optimizer = AdamW(params, lr=learning_rate, eps=adam_epsilon)
    loss_function = torch.nn.BCELoss(reduction='none')
    best_f1 = 0
    best_epoch = 0

    for i in range(configs.epoch):
        logger.info('epoch:{}/{}'.format(i + 1, configs.epoch))
        step, loss, loss_sum = 0, 0.0, 0.0
        for step, loader_res in tqdm(iter(enumerate(loader))):
            sentences = loader_res['sentence'].to(device)
            attention_mask = loader_res['attention_mask'].to(device)
            entity_vec = loader_res['entity_vec'].to(device)
            with torch.no_grad():
                bert_hidden_states = bert_model(sentences, attention_mask=attention_mask)[0].to(device)
            model_output = model(bert_hidden_states).to(device)
            loss = loss_function(model_output, entity_vec.float())
            loss = torch.sum(torch.mean(loss, 3), 2)
            loss = torch.sum(loss * attention_mask) / torch.sum(attention_mask)
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        logger.info('start evaluate engines...')
        results_of_each_entity = evaluate(configs, bert_model, model, dev_data, device)
        f1 = 0.0
        for class_id, performance in results_of_each_entity.items():
            f1 += performance['f1']
            # 打印每个类别的指标
            logger.info('class_name: %s, precision: %.4f, recall: %.4f, f1: %.4f'
                        % (class_id, performance['precision'], performance['recall'], performance['f1']))
        # 这里算得是所有类别的平均f1值
        f1 = f1 / len(results_of_each_entity)
        unprocessed = 0
        if f1 >= best_f1:
            unprocessed = 0
            best_f1 = f1
            best_epoch = i + 1
            model_name = 'model_' + str(i) + '.pkl'
            torch.save(model, os.path.join(configs.checkpoints_dir, model_name))
            logger.info('saved ' + model_name + ' successful...')
        else:
            unprocessed += 1
        aver_loss = loss_sum / step
        logger.info(
            'aver_loss: %.4f, f1: %.4f, best_f1: %.4f, best_epoch: %d \n' % (aver_loss, f1, best_f1, best_epoch))
        if configs.is_early_stop:
            if unprocessed > configs.patient:
                logger.info('early stopped, no progress obtained within {} epochs'.format(i + 1))
                break
