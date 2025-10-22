import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import pickle
import json
import os
import pandas as pd

# -------------------------------
# Config
# -------------------------------
MODEL_DIR = './models/classifier'
CACHE_FILE = './cache/feedback_cache.pkl'
CONFIDENCE_THRESHOLD = 0.7  # retrain if below
RETRAIN_BATCH_SIZE = 16
NUM_EPOCHS = 3

# -------------------------------
# Load model and tokenizer
# -------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

with open(os.path.join(MODEL_DIR, 'label_map.json'), 'r') as f:
    label_map = json.load(f)

reverse_label_map = {v: k for k, v in label_map.items()}

# -------------------------------
# Load or initialize feedback cache
# -------------------------------
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as f:
        feedback_cache = pickle.load(f)
else:
    feedback_cache = []  # List of dicts: {utterance, predicted_intent, confidence, feedback, correct_intent}

# -------------------------------
# Predict function
# -------------------------------
def predict(utterance):
    inputs = tokenizer(utterance, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence, pred_idx = torch.max(probs, dim=1)
    predicted_intent = reverse_label_map[pred_idx.item()]
    return predicted_intent, confidence.item()

# -------------------------------
# Add feedback to cache
# -------------------------------
def add_feedback(utterance, predicted_intent, confidence, feedback, correct_intent=None):
    feedback_cache.append({
        'utterance': utterance,
        'predicted_intent': predicted_intent,
        'confidence': confidence,
        'feedback': feedback,  # True/False
        'correct_intent': correct_intent
    })
    # Save cache
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(feedback_cache, f)

# -------------------------------
# Check if retraining is needed
# -------------------------------
def needs_retraining():
    for entry in feedback_cache:
        if not entry['feedback'] or entry['confidence'] < CONFIDENCE_THRESHOLD:
            return True
    return False

# -------------------------------
# Retrain model
# -------------------------------
def retrain_model():
    print("Retraining model with feedback data...")
    texts = []
    labels = []

    for entry in feedback_cache:
        if not entry['feedback'] and entry['correct_intent'] is not None:
            texts.append(entry['utterance'])
            labels.append(label_map[entry['correct_intent']])
        elif entry['feedback']:  # correct predictions above threshold
            texts.append(entry['utterance'])
            labels.append(label_map[entry['predicted_intent']])

    if not texts:
        print("No new data to retrain.")
        return

    # Create dataset
    import pandas as pd
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": texts, "labels": labels})

    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=RETRAIN_BATCH_SIZE,
        logging_dir='./logs',
        logging_steps=50,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset
    )

    trainer.train()

    # Save model
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # Clear cache after retraining
    feedback_cache.clear()
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(feedback_cache, f)

    print("Retraining complete. Cache cleared.")

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    while True:
        utterance = input("User: ")
        predicted_intent, confidence = predict(utterance)
        print(f"Bot Prediction: {predicted_intent} (Confidence: {confidence:.2f})")

        # Ask for user feedback
        feedback_input = input("Is this correct? (y/n): ").lower()
        if feedback_input == 'y':
            add_feedback(utterance, predicted_intent, confidence, True)
        else:
            correct_intent = input("Please provide the correct intent: ")
            add_feedback(utterance, predicted_intent, confidence, False, correct_intent)

        # Check if retraining is needed
        if needs_retraining():
            retrain_model()
