import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "intent_model"

class IntentPredictor:
    """Loads trained Hugging Face model and predicts intent."""
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded on device: {self.device}")

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confs, pred_ids = torch.max(probs, dim=1)

        results = []
        for i, text in enumerate(texts):
            intent = self.model.config.id2label[pred_ids[i].item()]
            confidence = confs[i].item()
            results.append((text, intent, confidence))
        return results


if __name__ == "__main__":
    predictor = IntentPredictor()
    if len(sys.argv) > 1:
        # CLI input
        text_input = " ".join(sys.argv[1:])
        results = predictor.predict(text_input)
    else:
        # Example batch input
        test_sentences = [
            "my flight got delayed",
            "how can I cancel my ticket?",
            "I want to change my seat",
            "is baggage included?",
            "my guitar case is broken"
        ]
        results = predictor.predict(test_sentences)

    for text, intent, conf in results:
        print(f"Text: '{text}'")
        print(f"Predicted Intent: '{intent}' (Confidence: {conf:.2f})")
