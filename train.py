from transformers import (
    BertForTokenClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

model = BertForTokenClassification.from_pretrained("cahya/bert-base-indonesian-NER")
tokenizer = BertTokenizer.from_pretrained("cahya/bert-base-indonesian-NER")
tokens = tokenizer.tokenize(
    "jl kapuk timur delta sili iii lippo cika 11 a cicau cikarang pusat"
)
ids = tokenizer.convert_tokens_to_ids(tokens)

print(tokens[:10], "...")
print(ids[:10], "...")
