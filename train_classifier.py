import transformers
import torch
from transformers import BertTokenizer, BertModel, DistilBertModel, DistilBertTokenizer
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

class sentenceData():
    def __init__(self, dataframe, bert_type):
        self.data = dataframe
        self.len = len(dataframe)
        self.bert_tokenizer_dict = {'BERT': BertTokenizer.from_pretrained('bert-base-uncased'),
                                    'DistilBert': BertTokenizer.from_pretrained('distilbert-base-uncased')}
        self.tokenizer = self.bert_tokenizer_dict[bert_type]
        self.max_len = 257

    def __getitem__(self, index):
        sentence = self.data['Text'][index]
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'targets': torch.tensor(self.data['int_label'][index], dtype=torch.long)
                }

    def visualization(self):
        sentence_length_list = []
        for sentence in self.data['SIT']:
            sentence_length_list.append(len(sentence.split(' ')))

        print(sentence_length_list)
        print(max(sentence_length_list))
        sns.set_style('darkgrid')
        sns.distplot(sentence_length_list)
        plt.xlim(1, 60)
        plt.show()
        labels = set(self.data['Field1'].tolist())
        print(labels)
        label_amount_dict = dict(zip(labels, [0]*len(labels)))
        print(label_amount_dict)
        for label in self.data['Field1']:
            label_amount_dict[label] += 1

        print(label_amount_dict)
        label_amount_df = pd.DataFrame({'emotion':label_amount_dict.keys(), 'amount':label_amount_dict.values()})
        print(label_amount_df)
        sns.barplot(x=label_amount_df['emotion'], y=label_amount_df['amount'], data=label_amount_df, ci=68)
        plt.show()

    def __len__(self):
        return self.len

class distil_bert_lstm(torch.nn.Module):
    def __init__(self, nclass):
        super(distil_bert_lstm, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.lstm = torch.nn.LSTM(
                input_size=768,
                batch_first=True,
                hidden_size=768,
                num_layers=1
            )
        self.classifier = torch.nn.Linear(768*256, nclass)

    def forward(self, input_ids, attention_mask):
        output_word = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_word[0]
        pooler = hidden_state[:, 1:].to(device)
        input_data = self.lstm(pooler)[0]
        input_data = input_data.reshape(input_data.size(0), -1)
        out = self.classifier(input_data)
        return out

def calculate_acc(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct

def train(epoch, model):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-05)
    model.train()
    for _, data in tqdm(enumerate(trainloader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)
        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calculate_acc(big_idx, targets)
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 50 == 0:
            loss_step = tr_loss / nb_tr_steps
            acc_step = (n_correct * 100) / nb_tr_examples
            print(f'Training Loss per 50 steps: {loss_step}, Training Accuracy per 50 steps: {acc_step}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Total Accuracy Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_acc = (n_correct * 100) / nb_tr_examples
    print(f'Training Loss Epoch: {epoch_loss}, Training Accuracy Epoch: {epoch_acc}')

    return

def valid(model, testloader):
    model.eval()
    n_correct = 0
    n_wrong = 0
    total = 0
    loss_function = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        tr_loss = 0
        nb_tr_steps = 1
        nb_tr_examples = 0
        targets_list = []
        label_pre_list = []
        for _, data in enumerate(testloader, 0):
            ids = data['ids'].to(device, dtype=torch.long)

            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            targets_list += targets
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            label_pre_list += big_idx
            n_correct += calculate_acc(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 100 == 0:
                loss_step = tr_loss / nb_tr_steps
                acc_step = (n_correct * 100) / nb_tr_examples
                print(f'Validation Loss per 100 steps: {loss_step}, Validation Accuracy per 100 steps: {acc_step}')

    epoch_loss = tr_loss / nb_tr_steps
    epoch_acc = (n_correct * 100) / nb_tr_examples
    print(f'Validation Loss Epoch: {epoch_loss}, Validation Accuracy Epoch: {epoch_acc}')

    return epoch_acc, targets_list, label_pre_list

if __name__ =='__main__':
    device = 'cuda'
    data = pd.read_csv(r'new_dataset.csv')
    train_size = 0.8
    train_dataset = data.sample(frac=train_size, random_state=20)
    test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    print(train_dataset.shape)
    print(test_dataset.shape)
    data_set = sentenceData(data, 'DistilBert')
    # data_set.visualization()
    training_set = sentenceData(train_dataset, 'DistilBert')
    testing_set = sentenceData(test_dataset, 'DistilBert')

    train_params = {'batch_size': 8,
                    'shuffle': True,
                    'num_workers': 0}
    test_params = {'batch_size': 8,
                   'shuffle': False,
                   'num_workers': 0}

    trainloader = DataLoader(training_set, **train_params)
    testloader = DataLoader(testing_set, **test_params)
    model = distil_bert_lstm(2)
    model.to(device)

    for epoch in range(2):
        train(epoch, model)

    acc, targets_list, label_pre_list = valid(model, testloader)
    print(acc)
    for index in range(len(targets_list)):
        targets_list[index] = targets_list[index].to('cpu').int()
        label_pre_list[index] = label_pre_list[index].to('cpu').int()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    conf_matrix = confusion_matrix(targets_list, label_pre_list)
    sns.heatmap(conf_matrix, annot=True, xticklabels=['Eng', 'Fre'],
                yticklabels=['Eng', 'Fre'])

    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.show()
    print(classification_report(targets_list, label_pre_list,
                                target_names=['Eng', 'Fre']))

    torch.save(model.state_dict(), 'classifierModel')


