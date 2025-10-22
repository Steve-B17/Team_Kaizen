import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# The "Champion" model is what we use for predictions
MODEL_PATH = "./intent_model"

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
        
        # Set up the device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode
        
        print(f"✅ IntentClassifier loaded successfully on device: {self.device}")

    def predict(self, text: str) -> str:
        """
        Predicts the intent for a single piece of text.
        """
        # Tokenize the input text and move tensors to the correct device
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        ).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Get the predicted class ID by finding the max logit
        predicted_id = torch.argmax(logits, dim=1).item()
        
        # Map the ID back to the human-readable label string
        return self.model.config.id2label[predicted_id]

# Create a single, global instance of the classifier
# This is loaded once when the application starts and reused for all predictions.
try:
    classifier = IntentClassifier()
except FileNotFoundError:
    classifier = None
    print("WARNING: Classifier could not be loaded. The /predict endpoint will not work.")