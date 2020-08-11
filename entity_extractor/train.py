from model import Model
from transformers import AdamW, BertModel
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import DataGenerator, MyDataset, collate_fn
from data import BATCH_SIZE, EPOCH_NUM
from predict import evaluate
from utils.logger import get_logger
import json
import torch

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger = get_logger('./log')
    train_data = json.load(open('./data/train_data.json', encoding='utf-8'))
    dev_data = json.load(open('./data/dev_data.json', encoding='utf-8'))
    data_generator = DataGenerator(train_data, logger)
    sentence, segment, attention_mask, entity_vectors = data_generator.prepare_data()
    torch_dataset = MyDataset(sentence, segment, attention_mask, entity_vectors)
    loader = DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # random shuffle for training
        num_workers=0,
        collate_fn=collate_fn,  # subprocesses for loading data
    )
    learning_rate = 4e-05
    adam_epsilon = 1e-05
    model = Model(hidden_size=768, num_labels=3).to(device)
    bert_model = BertModel.from_pretrained('bert-base-chinese').to(device)
    params = list(model.parameters())
    optimizer = AdamW(params, lr=learning_rate, eps=adam_epsilon)
    loss_function = torch.nn.BCELoss(reduction='none')
    best_f1 = 0
    best_epoch = 0

    for i in range(EPOCH_NUM):
        logger.info('epoch:{}/{}'.format(i, EPOCH_NUM))
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
            logger.info('loss in {} step:{}'.format(str(step), loss.item()))
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        logger.info('start evaluate model...\n')
        logger.info('dev_data_length:{}'.format(len(dev_data)))
        f1, precision, recall = evaluate(bert_model, model, dev_data, device)
        if f1 >= best_f1:
            best_f1 = f1
            best_epoch = i
            torch.save(model, './model/model_' + str(i) + '.pkl')
        logger.info('loss: %.4f, f1: %.4f, precision: %.4f, recall: %.4f, best_f1: %.4f, best_epoch: %d \n' % (
                loss_sum // step, f1, precision, recall, best_f1, best_epoch))
