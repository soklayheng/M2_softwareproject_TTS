import torch
from transformers import DistilBertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Distil_Bert_LSTM(torch.nn.Module):
    def __init__(self, nclass):
        super(Distil_Bert_LSTM, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # TODO: parametrize externally, unless works perfectly
        self.lstm = torch.nn.LSTM(
            input_size=768,
            batch_first=True,
            hidden_size=768,
            num_layers=1)
        self.classifier = torch.nn.Linear(768*256, nclass)

    def forward(self, input_ids, attention_mask):
        output_word = self.l1(input_ids=input_ids,
                              attention_mask=attention_mask)
        hidden_state = output_word[0]
        pooler = hidden_state[:, 1:].to(device)
        input_data = self.lstm(pooler)[0]
        input_data = input_data.reshape(input_data.size(0), -1)
        out = self.classifier(input_data)

        return out
