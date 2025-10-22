# In run_training_pipeline.py
import os
import json
import shutil

from train_model import train_model
from evaluate_model import evaluate_model

# --- Configuration ---
CHAMPION_MODEL_PATH = "./intent_model"
CHALLENGER_MODEL_PATH = "./challenger_model"
CHAMPION_METRICS_FILE = os.path.join(CHAMPION_MODEL_PATH, "evaluation_metrics.json")
PRIMARY_METRIC = "f1-score" # The key metric from 'weighted avg' to compare

def run_pipeline():
    """Manages the end-to-end training, evaluation, and deployment pipeline."""
    print("="*50)
    print("🚀 Starting new training and evaluation pipeline...")
    print("="*50)

    # --- 1. Train the Challenger Model ---
    # This will train the model and show the training/validation loss graphs upon completion.
    train_model(output_path=CHALLENGER_MODEL_PATH)

    # --- 2. Evaluate the Challenger Model ---
    print("\nEvaluating the new Challenger model...")
    challenger_metrics = evaluate_model(CHALLENGER_MODEL_PATH, show_plots=False)
    if not challenger_metrics:
        print("❌ Challenger evaluation failed. Aborting pipeline.")
        return
    
    challenger_score = challenger_metrics["weighted avg"][PRIMARY_METRIC]
    print(f"Challenger Score ({PRIMARY_METRIC}): {challenger_score:.4f}")

    # --- 3. Get the Champion Model's Score ---
    champion_score = 0.0
    if os.path.exists(CHAMPION_METRICS_FILE):
        with open(CHAMPION_METRICS_FILE, 'r') as f:
            champion_metrics = json.load(f)
        champion_score = champion_metrics["weighted avg"][PRIMARY_METRIC]
        print(f"Current Champion Score ({PRIMARY_METRIC}): {champion_score:.4f}")
    else:
        print("No existing Champion model found. The first model will be promoted.")

    # --- 4. Compare and Deploy or Discard ---
    if challenger_score > champion_score:
        print(f"\n✅ Challenger is better! ({challenger_score:.4f} > {champion_score:.4f})")
        print("Promoting Challenger to Champion...")
        
        if os.path.exists(CHAMPION_MODEL_PATH):
            shutil.rmtree(CHAMPION_MODEL_PATH)
        shutil.copytree(CHALLENGER_MODEL_PATH, CHAMPION_MODEL_PATH)
        
        with open(CHAMPION_METRICS_FILE, 'w') as f:
            json.dump(challenger_metrics, f, indent=4)
        
        print(f"🏆 New Champion model is live at '{CHAMPION_MODEL_PATH}'")

        # NEW: Display the full metrics graphs for the new Champion
        print("\nDisplaying final evaluation report for the new Champion...")
        evaluate_model(CHAMPION_MODEL_PATH, show_plots=True)

    else:
        print(f"\n❌ Challenger is not better. ({challenger_score:.4f} <= {champion_score:.4f})")
        print("Discarding Challenger and keeping the current Champion.")

    # --- 5. Clean up ---
    print("Cleaning up temporary challenger model directory...")
    shutil.rmtree(CHALLENGER_MODEL_PATH)
    
    print("\nPipeline finished.")
    print("="*50)

if __name__ == "__main__":
    run_pipeline()