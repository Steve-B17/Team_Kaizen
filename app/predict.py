# In app/predict.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./intent_model"

class IntentClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ IntentClassifier loaded on device: {self.device}")

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        predicted_id = torch.argmax(logits, dim=1).item()
        return self.model.config.id2label[predicted_id]

# Create a single instance to be used by the app
classifier = IntentClassifier()