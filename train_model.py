import os
import sys
import pandas as pd
import numpy as np
import torch
import pickle  # Added for .pkl saving
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from dotenv import load_dotenv

os.environ["WANDB_DISABLED"] = "true"
load_dotenv()  
# --- 1. Constants ---
MODEL_CHECKPOINT = "distilbert-base-uncased"
DB_SECRET_NAME = "POSTGRES_CONNECTION_STRING"
MODEL_OUTPUT_DIR = "./intent_model"
PICKLE_OUTPUT_FILE = "classifier.pkl" # File for the pickled model

# --- 2. Database Utilities ---
def get_db_engine():
    """Establishes a connection to the PostgreSQL database."""
    connection_string = os.environ.get(DB_SECRET_NAME)
    if not connection_string:
        print(f"❌ Database connection string not found. Set {DB_SECRET_NAME} in your .env file.", file=sys.stderr)
        sys.exit(1)
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
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

# --- 3. Metrics ---
def compute_metrics(eval_pred):
    """Computes accuracy, F1, precision, and recall for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# --- 4. Training ---
def train_model():
    """Main function to fetch data, train the model, and save it."""
    engine = get_db_engine()
    train_df = fetch_data_from_table(engine, "train_set")
    test_df = fetch_data_from_table(engine, "test_set")

    if train_df.empty or test_df.empty:
        print("❌ Cannot proceed with training. Check database tables.")
        return

    # Map labels to integers
    all_intents = sorted(train_df['intent'].astype(str).unique())
    label2id = {label: i for i, label in enumerate(all_intents)}
    id2label = {i: label for i, label in enumerate(all_intents)}
    num_labels = len(all_intents)
    print(f"Found {num_labels} unique intents.")

    # Convert pandas to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def preprocess_function(examples):
        """Tokenizes text and converts labels to IDs."""
        tokenized = tokenizer(examples['utterance'], padding=True, truncation=True)
        tokenized['label'] = [label2id[label] for label in examples['intent']]
        return tokenized

    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

    print("Loading model for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # --- MODIFIED SECTION ---
    # Training Arguments (compatible with older transformers)
    # Replaced 'evaluation_strategy' with 'do_eval'
    training_args = TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        logging_steps=10,
        do_eval=True, # <-- Use this argument for older versions
        save_total_limit=2
    )
    # --- END OF MODIFICATION ---

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("🚀 Starting training...")
    trainer.train()

    print("\n--- Final Model Evaluation ---")
    eval_results = trainer.evaluate()
    print(eval_results)

    # --- Standard Hugging Face Save Method (Recommended) ---
    print(f"\n✅ Training complete. Saving model to '{MODEL_OUTPUT_DIR}' (STANDARD METHOD)")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print("Model saved successfully in standard format.")

    # --- Pickle Save Method (As requested, but discouraged) ---
    print(f"\nSaving model as pickle file to '{PICKLE_OUTPUT_FILE}' (DISCOURAGED METHOD)")
    print("WARNING: This .pkl file is not secure and will not work with your IntentPredictor class.")
    with open(PICKLE_OUTPUT_FILE, "wb") as f:
        pickle.dump(trainer.model, f)
    print(f"Model saved to {PICKLE_OUTPUT_FILE}.")


class IntentPredictor:
    """Class to load the saved model and run predictions."""
    def _init_(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}. Run training first.")
        print(f"Loading model from {model_path}...")
        
        # This loads the standard Hugging Face format (not the .pkl)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"✅ Predictor loaded. Using device: {self.device}")

    def predict(self, texts):
        """Batch prediction for one or more sentences."""
        if isinstance(texts, str):
            texts = [texts]
            
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confs, pred_ids = torch.max(probs, dim=1)

        results = []
        for i in range(len(texts)):
            intent = self.model.config.id2label[pred_ids[i].item()]
            confidence = confs[i].item()
            results.append((texts[i], intent, confidence))
            
        return results

# --- 6. Main Execution ---
if __name__ == "__main__":
    # Step 1: Train and save the model
    train_model()

    # Step 2: Load the saved model (from MODEL_OUTPUT_DIR) and test it
    print("\n--- Testing Predictions ---")
    try:
        predictor = IntentPredictor(MODEL_OUTPUT_DIR)
        test_sentences = [
            "my bag didn't show up at JFK",
            "i need to reschedule my flight",
            "how much is a ticket to tokyo?",
            "what's the pet policy?",
            "my guitar case is broken",
            "is my flight on time?"
        ]
        results = predictor.predict(test_sentences)
        
        for text, intent, conf in results:
            print(f"Text: '{text}'")
            print(f"Predicted Intent: '{intent}' (Confidence: {conf:.2f})\n")
            
    except FileNotFoundError as e:
        print(f"Skipping prediction — model not found. {e}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")