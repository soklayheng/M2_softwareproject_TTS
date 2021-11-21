
import transformers
import torch
from transformers import BertTokenizer, BertModel, DistilBertModel, DistilBertTokenizer
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

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
        pooler = hidden_state[:, 1:].to('cuda')
        input_data = self.lstm(pooler)[0]
        input_data = input_data.reshape(input_data.size(0), -1)
        out = self.classifier(input_data)
        return out

sentenceInput = input()
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
inputData = tokenizer.encode_plus(
    sentenceInput,
    None,
    add_special_tokens=True,
    max_length=257,
    padding='max_length',
    return_token_type_ids=True,
    truncation=True
)
ids = torch.tensor([inputData['input_ids']], dtype=torch.long).to('cuda')
mask = torch.tensor([inputData['attention_mask']], dtype=torch.long).to('cuda')

classifierModel = distil_bert_lstm(2).to('cuda')
classifierModel.load_state_dict(torch.load(r'classifierModel'))
outputs = classifierModel(ids, mask)
result = torch.max(outputs.data, dim=1).indices
EN = torch.tensor([1]).to('cuda')

if result == EN:
    result = 'EN'
else:
    result = 'FR'

print(result)
