import os
import sys
import argparse
import subprocess

reqs = ["transformers", "datasets", "torch", "scikit-learn", "joblib", "pandas"]
for p in reqs:
    try:
        __import__(p)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", p])

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="cleaned_dataset.csv")
parser.add_argument("--output_dir", default="./sentiment_model")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_length", type=int, default=128)
args = parser.parse_args()

if not os.path.exists(args.dataset):
    print("Dataset not found:", args.dataset)
    sys.exit(1)

df = pd.read_csv(args.dataset)
if 'text' not in df.columns:
    possible = [c for c in df.columns if 'text' in c.lower()]
    if possible:
        df.rename(columns={possible[0]: 'text'}, inplace=True)
if 'sentiment' not in df.columns:
    possible = [c for c in df.columns if 'sent' in c.lower()]
    if possible:
        df.rename(columns={possible[0]: 'sentiment'}, inplace=True)
if 'text' not in df.columns or 'sentiment' not in df.columns:
    print('Required columns missing: need text and sentiment')
    sys.exit(1)

df = df.dropna(subset=['text'])
df['sentiment'] = df['sentiment'].astype(str)
labels = sorted(df['sentiment'].unique())
label2id = {lab: i for i, lab in enumerate(labels)}
id2label = {i: lab for lab, i in label2id.items()}
df['label'] = df['sentiment'].map(label2id)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)
train_df = pd.DataFrame({'text': X_train, 'label': y_train.values})
test_df = pd.DataFrame({'text': X_test, 'label': y_test.values})
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=args.max_length)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=len(labels)
)

# ✅ Compatible with old and new transformers versions
try:
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
    )
except TypeError:
    # Fallback for older versions
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

trainer.train()

preds = trainer.predict(test_dataset)
y_pred = np.argmax(preds.predictions, axis=1)
y_true = test_df['label'].values
report = classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(len(labels))])
print(report)

os.makedirs(args.output_dir, exist_ok=True)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
joblib.dump({'label2id': label2id, 'id2label': id2label}, os.path.join(args.output_dir, 'label_maps.pkl'))

try:
    joblib.dump(model, os.path.join(args.output_dir, 'sentiment_model.pkl'))
except Exception:
    pass

with open('requirements.txt', 'w') as f:
    f.write('\\n'.join(reqs))

print('Artifacts saved to', args.output_dir)
print('requirements.txt written')
