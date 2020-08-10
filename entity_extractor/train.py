from model import Model
from transformers import AdamW, BertModel
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import DataGenerator, MyDataset, collate_fn
from data import BATCH_SIZE, EPOCH_NUM
from predict import evaluate
import json
import torch

if __name__ == '__main__':
    train_data = json.load(open('./data/train_data.json', encoding='utf-8'))
    dev_data = json.load(open('./data/dev_data.json', encoding='utf-8'))
    data_generator = DataGenerator(train_data)
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
    model = Model(hidden_size=768, num_labels=3)
    bert_model = BertModel.from_pretrained('bert-base-chinese')
    params = list(model.parameters())
    optimizer = AdamW(params, lr=learning_rate, eps=adam_epsilon)
    loss_function = torch.nn.BCELoss(reduction='none')
    loss, loss_sum = 0, 0
    best_f1 = 0
    best_epoch = 0

    for i in range(EPOCH_NUM):
        print('epoch:{}/{}'.format(i, EPOCH_NUM))
        for step, loader_res in tqdm(iter(enumerate(loader))):
            sentences = loader_res['sentence']
            attention_mask = loader_res['attention_mask']
            entity_vec = loader_res['entity_vec']
            with torch.no_grad():
                bert_hidden_states, batch_attentions = bert_model(sentences, attention_mask=attention_mask)
            model_output = model(bert_hidden_states)
            loss = loss_function(model_output, entity_vec.float())
            loss = torch.sum(torch.mean(loss, 3), 2)
            loss = torch.sum(loss * attention_mask) / torch.sum(attention_mask)
            print('loss:{}'.format(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        f1, precision, recall = evaluate(bert_model, model, dev_data)
        if f1 >= best_f1:
            best_f1 = f1
            best_epoch = i

        print('f1: %.4f, precision: %.4f, recall: %.4f, bestf1: %.4f, bestepoch: %d \n ' % (
            f1, precision, recall, best_f1, best_epoch))