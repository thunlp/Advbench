import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer




class processed_dataset(Dataset):
    def __init__(self, data, bert_type):
        tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.texts = []
        self.labels = []
        for text, label in data:
            self.texts.append(torch.tensor(tokenizer.encode(text, max_length=512, truncation=True)))
            self.labels.append(label)
        assert len(self.texts) == len(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class packDataset_util():
    def __init__(self, bert_type):
        self.bert_type = bert_type

        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
    def fn(self, data):
        texts = []
        labels = []
        for text, label in data:
            texts.append(text)
            labels.append(label)
        labels = torch.tensor(labels)
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_masks = torch.zeros_like(padded_texts).masked_fill(padded_texts != self.tokenizer.pad_token_id, 1)
        return padded_texts, attention_masks, labels


    def get_loader(self, data, shuffle=True, batch_size=32):
        dataset = processed_dataset(data, self.bert_type)
        loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=self.fn)
        return loader
