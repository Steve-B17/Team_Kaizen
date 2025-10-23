# In evaluate_model.py
import os
import sys
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

load_dotenv()
# --- Helper Functions (Keep these as they are) ---
# --- Constants ---
DB_SECRET_NAME = "POSTGRES_CONNECTION_STRING"
MODEL_PATH = "./intent_model"

# --- Database Utilities ---
def get_db_engine():
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
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        print(f"✅ Fetched {len(df)} rows from '{table_name}'.")
        return df
    except Exception as e:
        print(f"❌ Failed to read table '{table_name}': {e}", file=sys.stderr)
        return pd.DataFrame()

# --- NEW: Plotting Function for Classification Report ---
def plot_classification_report(report):
    """Plots Precision, Recall, and F1-score from a classification report."""
    # Convert report to a pandas DataFrame
    report_df = pd.DataFrame(report).transpose()
    # Get the class labels (intents), excluding avg/total rows
    class_labels = list(report_df.index[:-3])
    
    # Isolate the metrics for plotting
    metrics_df = report_df.loc[class_labels, ['precision', 'recall', 'f1-score']]
    
    ax = metrics_df.plot(kind='bar', figsize=(12, 7))
    plt.title('Performance Metrics per Intent')
    plt.ylabel('Score')
    plt.xlabel('Intents')
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--')
    plt.legend(loc='lower right')
    plt.tight_layout()

# --- Main Evaluation Function ---
# In evaluate_model.py

def evaluate_model(model_path, show_plots=False):
    """Evaluates a trained model against the test set."""
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}", file=sys.stderr)
        return None

    # --- 1. Load Model and Tokenizer ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully from '{model_path}' on device: {device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}", file=sys.stderr)
        return None

    # --- 2. Load Test Data ---
    engine = get_db_engine() # Make sure you have this helper function in the file
    test_df = fetch_data_from_table(engine, "test_set") # Make sure you have this one too
    if test_df.empty:
        print("❌ Test data not found or is empty.", file=sys.stderr)
        return None

    # --- 3. Generate Predictions (THIS WAS THE MISSING PART) ---
    true_labels = []
    predicted_labels = []
    
    print("Running predictions on the test set...")
    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        text = row['utterance']
        true_label = row['intent']
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        predicted_id = torch.argmax(logits, dim=1).item()
        predicted_label = model.config.id2label[predicted_id]
        
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    # --- 4. Metrics Calculation and Visualization ---
    print(f"\n--- Evaluating Model at: {model_path} ---")
    
    # Get all unique labels from the test set for the report
    labels = sorted(list(pd.unique(test_df['intent'].tolist())))
    
    report_dict = classification_report(true_labels, predicted_labels, labels=labels, digits=4, output_dict=True)
    print(classification_report(true_labels, predicted_labels, labels=labels, digits=4))
    
    if show_plots:
        # Create and display the metrics bar chart
        plot_classification_report(report_dict)

        # Create and display the confusion matrix
        fig_cm, ax_cm = plt.subplots(figsize=(10, 10))
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues, ax=ax_cm, xticks_rotation='vertical')
        ax_cm.set_title("Confusion Matrix", fontsize=16)
        plt.tight_layout()

        plt.show()

    return report_dict

if __name__ == "__main__":
    evaluate_model("./intent_model", show_plots=True)