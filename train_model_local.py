import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import json
import os

# -------------------------------
# Config
# -------------------------------
DATA_FILE = './data/synthetic_data.csv'  # CSV with columns: utterance,intent
MODEL_DIR = './models/classifier'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(DATA_FILE)
texts = df['utterance'].tolist()
intents = df['intent'].tolist()

# Create label map
unique_intents = sorted(list(set(intents)))
label_map = {intent: idx for idx, intent in enumerate(unique_intents)}
labels = [label_map[intent] for intent in intents]

# Create Hugging Face dataset
dataset = Dataset.from_dict({"text": texts, "labels": labels})

# -------------------------------
# Tokenizer
# -------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# -------------------------------
# Model
# -------------------------------
num_labels = len(unique_intents)
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=num_labels
)

# -------------------------------
# Training arguments (compatible with older versions)
# -------------------------------
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_dir='./logs',
    logging_steps=50,
    save_total_limit=1  # Keeps only last checkpoint
)

# -------------------------------
# Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset  # Using same dataset for eval; fine for small synthetic dataset
)

# -------------------------------
# Train
# -------------------------------
trainer.train()

# -------------------------------
# Save model, tokenizer, label map
# -------------------------------
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

with open(os.path.join(MODEL_DIR, 'label_map.json'), 'w') as f:
    json.dump(label_map, f)

print(f"Training complete. Model and tokenizer saved to {MODEL_DIR}")
