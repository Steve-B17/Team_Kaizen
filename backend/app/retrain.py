import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Setup ---
os.environ["WANDB_DISABLED"] = "true"
load_dotenv()

# --- Constants ---
MODEL_CHECKPOINT = "distilbert-base-uncased"
DB_SECRET_NAME = "POSTGRES_CONNECTION_STRING"

# --- Database Utilities ---
def get_db_engine():
    """Establishes a connection to the PostgreSQL database."""
    connection_string = os.environ.get(DB_SECRET_NAME)
    if not connection_string:
        print(f"❌ Set {DB_SECRET_NAME} in your .env file.", file=sys.stderr)
        sys.exit(1)
    try:
        engine = create_engine(connection_string)
        print("✅ Successfully connected to PostgreSQL Database!")
        return engine
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}", file=sys.stderr)
        sys.exit(1)

def fetch_data_from_table(engine, table_name):
    """Fetches data from a specific table into a pandas DataFrame."""
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        print(f"✅ Fetched {len(df)} rows from '{table_name}'.")
        return df
    except Exception as e:
        print(f"❌ Failed to read table '{table_name}': {e}", file=sys.stderr)
        return pd.DataFrame()

# --- Metrics ---
def compute_metrics(eval_pred):
    """Computes metrics for the validation set during training."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# --- Plotting Function ---
def plot_training_history(history):
    """Plots the training and validation loss and accuracy from the trainer history."""
    eval_logs = [log for log in history if 'eval_loss' in log]
    train_logs = [log for log in history if 'loss' in log and 'eval_loss' not in log]
    
    if not eval_logs:
        print("No evaluation logs found. Cannot plot history.")
        return

    # Align training and eval logs by epoch
    eval_epochs = [log['epoch'] for log in eval_logs]
    eval_loss = [log['eval_loss'] for log in eval_logs]
    eval_accuracy = [log['eval_accuracy'] for log in eval_logs]
    
    # Get the training loss from the step closest to the eval epoch
    train_loss_map = {log['epoch']: log['loss'] for log in train_logs}
    train_loss = [train_loss_map.get(epoch, None) for epoch in eval_epochs]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    
    # Filter out None values if training loss wasn't logged at the exact eval epoch
    valid_train_epochs = [e for e, t in zip(eval_epochs, train_loss) if t is not None]
    valid_train_loss = [t for t in train_loss if t is not None]
    
    if valid_train_loss:
        plt.plot(valid_train_epochs, valid_train_loss, 'o-', label='Training Loss')
        
    plt.plot(eval_epochs, eval_loss, 'o-', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(eval_epochs, eval_accuracy, 'o-', label='Validation Accuracy', color='orange')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# --- Main Training Function ---
def train_model(output_path="./challenger_model"):
    """
    Fetches the latest training data AND feedback, combines them,
    and trains a new challenger model.
    """
    # 1. Fetch and Combine Data
    print("🚀 Starting retraining pipeline...")
    engine = get_db_engine()
    
    # Fetch original training data
    original_train_df = fetch_data_from_table(engine, "train_set")
    if original_train_df.empty:
        print("❌ Original training data not found. Aborting.", file=sys.stderr)
        return None
    
    # Fetch only the CORRECTED mistakes from feedback
    print("Fetching new feedback...")
    feedback_df = pd.read_sql(
        "SELECT utterance, correct_intent AS intent FROM feedback_logs WHERE is_correct = false",
        engine
    )
    
    if feedback_df.empty:
        print("ℹ️ No new incorrect feedback to train on. Training on original data only.")
        combined_df = original_train_df
    else:
        print(f"✅ Fetched {len(original_train_df)} original rows and {len(feedback_df)} new rows from feedback.")
        # Combine datasets and remove duplicates, keeping the feedback (last) as truth
        combined_df = pd.concat([original_train_df, feedback_df], ignore_index=True)
        combined_df.drop_duplicates(subset=['utterance'], keep='last', inplace=True)
        print(f"Total combined rows after deduplication: {len(combined_df)}")

    # 2. Prepare Data for Training
    train_df, val_df = train_test_split(
        combined_df, test_size=0.1, random_state=42, stratify=combined_df['intent']
    )

    all_intents = sorted(combined_df["intent"].astype(str).unique())
    label2id = {label: i for i, label in enumerate(all_intents)}
    id2label = {i: label for i, label in enumerate(all_intents)}
    num_labels = len(all_intents)

    # 3. Load Tokenizer and Preprocess Data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def preprocess_function(examples):
        tokenized = tokenizer(examples["utterance"], padding=True, truncation=True)
        tokenized["label"] = [label2id[label] for label in examples["intent"]]
        return tokenized

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)

    # 4. Load Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # 5. Define Training Arguments and Trainer
    training_args = TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        logging_strategy="epoch",  # Log at the end of each epoch
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch",     # Save at the end of each epoch
        load_best_model_at_end=True, # Load the best model at the end of training
        save_total_limit=2,
        metric_for_best_model="f1", # Use F1 score to determine the best model
        report_to="none", # Disables wandb logging
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # 6. Train and Save
    print("🚀 Starting training for the Challenger model...")
    trainer.train()

    print("\nDisplaying training graphs...")
    plot_training_history(trainer.state.log_history)

    print(f"\n✅ Training complete. Saving best model to '{output_path}'")
    trainer.save_model(output_path)
    
    label_map_path = os.path.join(output_path, "id2label.json")
    with open(label_map_path, 'w') as f:
        json.dump(id2label, f, indent=2)
    print(f"Label map saved to '{label_map_path}'")
    
    return output_path

if __name__ == "__main__":
    # This will train and save the model to "./challenger_model"
    # which is what `run_training_pipeline.py` expects.
    train_model()