# from tqdm import tqdm

import pandas as pd
# from matplotlib import pyplot as plt
# import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from models.disitlbert import distil_bert_lstm


class SentenceData():
    def __init__(self, dataframe, bert_type):
        self.data = dataframe
        self.len = len(dataframe)
        self.bert_tokenizer_dict = {
            "BERT": BertTokenizer.from_pretrained("bert-base-uncased"),
            "DistilBert": BertTokenizer.from_pretrained("distilbert-base-uncased")}
        self.tokenizer = self.bert_tokenizer_dict[bert_type]
        self.max_len = 257

    def __getitem__(self, index):
        sentence = self.data["Text"][index]
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "targets": torch.tensor(self.data["int_label"][index], dtype=torch.long)}

    # TODO: not used visualization; delete, unless useful for report
    # def visualization(self):
    #     sentence_length_list = [len(sent.split(" "))
    #                             for sent in self.data["SIT"]]

    #     sns.set_style("darkgrid")
    #     sns.distplot(sentence_length_list)
    #     plt.xlim(1, 60)
    #     plt.show()

    #     labels = set(self.data["Field1"].tolist())
    #     label_amount_dict = dict(zip(labels, [0]*len(labels)))
    #     for label in self.data["Field1"]:
    #         label_amount_dict[label] += 1

    #     label_amount_df = pd.DataFrame({
    #         "emotion": label_amount_dict.keys(),
    #         "amount": label_amount_dict.values()})
    #     sns.barplot(
    #         x=label_amount_df["emotion"],
    #         y=label_amount_df["amount"],
    #         data=label_amount_df,
    #         ci=67)
    #     plt.show()

    def __len__(self):
        return self.len


def calculate_acc(big_idx, targets):
    return (big_idx == targets).sum().item()


def train(epoch, model):
    tr_loss, n_correct, nb_tr_steps, nb_tr_examples = 0, 0, 0, 0
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-05)

    model.train()

    # for i, data in tqdm(enumerate(trainloader)):
    for i, data in enumerate(trainloader):
        ids = data["ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)
        targets = data["targets"].to(device, dtype=torch.long)
        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        _, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calculate_acc(big_idx, targets)
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if i % 5 == 0:
            loss_step = tr_loss / nb_tr_steps
            acc_step = (n_correct * 100) / nb_tr_examples
            print(
                f"Training Loss per 50 steps: {loss_step}, Training Accuracy per 50 steps: {acc_step}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(
        f"Total Accuracy Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}")
    epoch_loss = tr_loss / nb_tr_steps
    epoch_acc = (n_correct * 100) / nb_tr_examples
    print(
        f"Training Loss Epoch: {epoch_loss}, Training Accuracy Epoch: {epoch_acc}")

    return None


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

        for i, data in enumerate(testloader):
            ids = data["ids"].to(device, dtype=torch.long)

            mask = data["mask"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.long)
            targets_list += targets
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            _, big_idx = torch.max(outputs.data, dim=1)
            label_pre_list += big_idx
            n_correct += calculate_acc(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if i % 100 == 0:
                loss_step = tr_loss / nb_tr_steps
                acc_step = (n_correct * 100) / nb_tr_examples
                print(
                    f"Validation Loss per 100 steps: {loss_step}, Validation Accuracy per 100 steps: {acc_step}")

    epoch_loss = tr_loss / nb_tr_steps
    epoch_acc = (n_correct * 100) / nb_tr_examples
    print(
        f"Validation Loss Epoch: {epoch_loss}, Validation Accuracy Epoch: {epoch_acc}")

    return epoch_acc, targets_list, label_pre_list


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv("C:/Users/shali/my_git_repos/M2_softwareproject_TTS/data/lang_dataset.csv")
    train_size = 0.8
    train_dataset = data.sample(frac=train_size, random_state=20)
    test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print(f"Train size: {train_dataset.shape}")
    print(f"Test size: {test_dataset.shape}")

    # data_set = SentenceData(data, "DistilBert")
    # data_set.visualization()

    training_set = SentenceData(train_dataset, "DistilBert")
    testing_set = SentenceData(test_dataset, "DistilBert")

    trainloader = DataLoader(
        training_set, **{"batch_size": 8, "shuffle": True, "num_workers": 0})
    testloader = DataLoader(
        testing_set, **{"batch_size": 8, "shuffle": False, "num_workers": 0})

    model = distil_bert_lstm(2)
    model.to(device)

    for epoch in range(2):
        train(epoch, model)

    acc, targets_list, label_pre_list = valid(model, testloader)
    print(f"accuracy: {acc}")

    for index in range(len(targets_list)):
        targets_list[index] = targets_list[index].to("cpu").int()
        label_pre_list[index] = label_pre_list[index].to("cpu").int()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # conf_matrix = confusion_matrix(targets_list, label_pre_list)
    # sns.heatmap(
    #     conf_matrix, annot=True, xticklabels=["Eng", "Fre"],
    #     yticklabels=["Eng", "Fre"])

    # ax.set_xlabel("predict")
    # ax.set_ylabel("true")
    # plt.show()
    print(classification_report(
        targets_list, label_pre_list,
        target_names=["Eng", "Fre"]))

    torch.save(model.state_dict(), "classifierModel")

    # TODO: add prediction
