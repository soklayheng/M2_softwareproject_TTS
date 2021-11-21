import torch
from transformers import BertTokenizer, DistilBertModel

from models.disitlbert import distil_bert_lstm


tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifierModel = distil_bert_lstm(2).to(device)
classifierModel.load_state_dict(torch.load(r"classifierModel"))


def classify(sentenceInput):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # sentenceInput = input()
    inputData = tokenizer.encode_plus(
        sentenceInput,
        None,
        add_special_tokens=True,
        max_length=257,
        padding="max_length",
        return_token_type_ids=True,
        truncation=True)

    ids = torch.tensor([inputData["input_ids"]], dtype=torch.long).to(device)
    mask = torch.tensor([inputData["attention_mask"]],
                        dtype=torch.long).to(device)

    outputs = classifierModel(ids, mask)
    result = torch.max(outputs.data, dim=1).indices
    EN = torch.tensor([1]).to(device)

    result = "EN" if result == EN else "FR"

    return result
