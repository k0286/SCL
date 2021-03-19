import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    BertForTokenClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
)

df = pd.read_csv("data/train_ner.csv")
# texts=df['raw_address'].str.split().to_list()
texts = (
    df["raw_addr_tokens"]
    .str.replace("'", "")
    .str.replace(" ", "")
    .apply(lambda x: x[1:-1].split(","))
    .to_list()
)
tags = (
    df["label"]
    .str.replace("'", "")
    .str.replace(" ", "")
    .apply(lambda x: x[1:-1].split(","))
    .to_list()
)

train_texts, val_texts, train_tags, val_tags = train_test_split(
    texts, tags, test_size=0.2
)
unique_tags = set(tag for doc in tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}


tokenizer = BertTokenizerFast.from_pretrained("cahya/bert-base-indonesian-NER")
train_encodings = tokenizer(
    train_texts,
    is_split_into_words=True,
    return_offsets_mapping=True,
    padding=True,
    truncation=True,
)
val_encodings = tokenizer(
    val_texts,
    is_split_into_words=True,
    return_offsets_mapping=True,
    padding=True,
    truncation=True,
)


def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset, ids in zip(
        labels, encodings.offset_mapping, encodings.input_ids
    ):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels


train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)


class NerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_encodings.pop("offset_mapping")  # we don't want to pass this to the model
val_encodings.pop("offset_mapping")

train_dataset = NerDataset(train_encodings, train_labels)
val_dataset = NerDataset(val_encodings, val_labels)

model = BertForTokenClassification.from_pretrained("cahya/bert-base-indonesian-NER")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


training_args = TrainingArguments(
    output_dir="./results",  # output directory
    num_train_epochs=30,  # total # of training epochs
    per_device_train_batch_size=128,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
