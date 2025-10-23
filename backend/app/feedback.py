import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Setup ---
load_dotenv()

# --- Constants ---
DB_SECRET_NAME = "POSTGRES_CONNECTION_STRING"
MODEL_PATH = "./intent_model"

# --- Database Utility ---
def get_db_engine():
    """Establishes a connection to the PostgreSQL database."""
    connection_string = os.environ.get(DB_SECRET_NAME)
    if not connection_string:
        print(f"❌ Set {DB_SECRET_NAME} in your .env file.", file=sys.stderr)
        sys.exit(1)
    try:
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}", file=sys.stderr)
        sys.exit(1)

def log_feedback(
    utterance: str, 
    predicted_intent: str, 
    is_correct: bool, 
    correct_intent: Optional[str] = None
):
    """
    Logs the user's feedback into the 'feedback_logs' table in the database.
    """
    engine = get_db_engine()
    
    query = text(
        """
        INSERT INTO feedback_logs (
            utterance, 
            predicted_intent, 
            is_correct, 
            correct_intent,
            created_at
        )
        VALUES (
            :utterance, 
            :predicted_intent, 
            :is_correct, 
            :correct_intent,
            NOW()
        )
        """
    )
    
    try:
        with engine.connect() as connection:
            connection.execute(query, {
                "utterance": utterance,
                "predicted_intent": predicted_intent,
                "is_correct": is_correct,
                "correct_intent": correct_intent
            })
            connection.commit()
        print("✅ Feedback logged successfully.")
        return True
    except Exception as e:
        print(f"❌ Error logging feedback: {e}")
        return False

def get_available_intents_from_model() -> List[str]:
    """
    Retrieves all available intent labels from the trained model.
    This is useful for providing dropdown options in the UI.
    """
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️ Model path not found at {MODEL_PATH}")
            return []
        
        # Load model config to get intent labels
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(MODEL_PATH)
        
        if hasattr(config, 'id2label'):
            intents = list(config.id2label.values())
            return sorted(intents)
        else:
            print("⚠️ Model config doesn't have id2label mapping")
            return []
    except Exception as e:
        print(f"❌ Error retrieving intents: {e}")
        return []

def collect_feedback_interactive(utterance: str, predicted_intent: str, available_intents: Optional[List[str]] = None):
    """
    Interactively collects feedback from the user.
    If the prediction is wrong, asks for the correct intent.
    Optionally displays available intents if provided.
    """
    print(f"\n📝 Utterance: '{utterance}'")
    print(f"🤖 Predicted Intent: '{predicted_intent}'")
    print("\nIs this prediction correct? (yes/no): ", end="")
    
    response = input().strip().lower()
    
    if response in ['yes', 'y', 'correct', 'right']:
        # Prediction is correct
        success = log_feedback(utterance, predicted_intent, is_correct=True)
        if success:
            print("✅ Thank you for confirming!")
        return True
    
    elif response in ['no', 'n', 'wrong', 'incorrect']:
        # Prediction is wrong - ask for correct intent
        print("\n❌ Sorry the prediction was incorrect.")
        
        # Show available intents if provided
        if available_intents:
            print("\n📋 Available intents:")
            for idx, intent in enumerate(available_intents, 1):
                print(f"  {idx}. {intent}")
            print("\nYou can enter the intent name or number from the list above.")
        
        print("\nPlease provide the correct intent: ", end="")
        
        correct_intent_input = input().strip()
        
        if correct_intent_input:
            # Check if user entered a number (selecting from list)
            if available_intents and correct_intent_input.isdigit():
                intent_idx = int(correct_intent_input) - 1
                if 0 <= intent_idx < len(available_intents):
                    correct_intent = available_intents[intent_idx]
                else:
                    print("⚠️ Invalid number. Using your input as-is.")
                    correct_intent = correct_intent_input
            else:
                correct_intent = correct_intent_input
            
            success = log_feedback(
                utterance=utterance,
                predicted_intent=predicted_intent,
                is_correct=False,
                correct_intent=correct_intent
            )
            
            if success:
                print(f"✅ Thank you! Logged correct intent as: '{correct_intent}'")
            return False
        else:
            print("⚠️ No correct intent provided. Logging as incorrect without correction.")
            log_feedback(utterance, predicted_intent, is_correct=False)
            return False
    
    else:
        print("⚠️ Invalid response. Please answer 'yes' or 'no'.")
        return collect_feedback_interactive(utterance, predicted_intent, available_intents)

# --- Intent Classifier ---
class IntentClassifier:
    def __init__(self, model_path=MODEL_PATH):
        """
        Loads the trained model and tokenizer from the specified path.
        """
        if not os.path.exists(model_path):
            print(f"❌ Error: Model path not found at {model_path}")
            print("Please run the training pipeline first to create a model.")
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ IntentClassifier loaded successfully on device: {self.device}")

    def predict(self, text: str) -> str:
        """
        Predicts the intent for a single piece of text.
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        predicted_id = torch.argmax(logits, dim=1).item()
        
        return self.model.config.id2label[predicted_id]
    
    def get_available_intents(self) -> List[str]:
        """
        Returns a sorted list of all available intent labels from the model.
        """
        return sorted(list(self.model.config.id2label.values()))
    
    def predict_with_feedback(self, text: str, collect_feedback: bool = True) -> str:
        """
        Predicts intent and optionally collects user feedback.
        Shows available intents when asking for corrections.
        """
        predicted_intent = self.predict(text)
        
        if collect_feedback:
            available_intents = self.get_available_intents()
            collect_feedback_interactive(text, predicted_intent, available_intents)
        
        return predicted_intent

# Initialize the classifier
try:
    classifier = IntentClassifier()
except FileNotFoundError:
    classifier = None
    print("WARNING: Classifier could not be loaded. Predictions will not work.")

# --- Example Usage ---
if __name__ == "__main__":
    if classifier is None:
        print("❌ Cannot run examples without a trained model.")
        sys.exit(1)
    
    # Example 1: Get available intents
    print("\n" + "="*60)
    print("AVAILABLE INTENTS")
    print("="*60)
    intents = classifier.get_available_intents()
    for i, intent in enumerate(intents, 1):
        print(f"{i}. {intent}")
    
    # Example 2: Predict with interactive feedback
    test_utterances = [
        "I want to book a flight to Paris",
        "What's the weather like today?",
        "Cancel my reservation"
    ]
    
    print("\n" + "="*60)
    print("INTENT CLASSIFICATION WITH FEEDBACK COLLECTION")
    print("="*60)
    
    for utterance in test_utterances:
        classifier.predict_with_feedback(utterance, collect_feedback=True)
        print("\n" + "-"*60)