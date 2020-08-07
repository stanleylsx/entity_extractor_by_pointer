from model import Model
from transformers import AdamW, BertModel
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import DataGenerator, MyDataset, collate_fn
from data import BATCH_SIZE, EPOCH_NUM
import json
import torch


if __name__ == '__main__':
    train_data = json.load(open('./data/train_data.json', encoding='utf-8'))
    # dev_data = json.load(open('data/dev_data.json'))
    data_generator = DataGenerator(train_data)
    sentence, start_vectors, end_vectors = data_generator.prepare_data()
    torch_dataset = MyDataset(sentence, start_vectors, end_vectors)
    loader = DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # random shuffle for training
        num_workers=0,
        collate_fn=collate_fn,  # subprocesses for loading data
    )
    learning_rate = 4e-05
    adam_epsilon = 1e-05
    model = Model()
    bert_model = BertModel.from_pretrained('bert-base-chinese')
    params = list(model.parameters())
    optimizer = AdamW(params, lr=learning_rate, eps=adam_epsilon)
    loss = torch.nn.CrossEntropyLoss()

    for i in range(EPOCH_NUM):
        for step, loader_res in tqdm(iter(enumerate(loader))):
            sentences = loader_res['sentence']
            start_vec = loader_res['start_vec']
            end_vec = loader_res['end_vec']
            with torch.no_grad():
                batch_hidden_states, batch_attentions = bert_model(sentences)[-2:]
            start, end = model(batch_hidden_states)
            start = start.permute(0, 2, 1)
            end = end.permute(0, 2, 1)
            s_loss = loss(start, start_vec)
            e_loss = loss(end, end_vec)

            loss_sum = s_loss + e_loss
            print(loss_sum)
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()