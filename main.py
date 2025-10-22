# In main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Import your existing classifier and feedback logger
from app.predict import classifier
from app.feedback import log_feedback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create the FastAPI app instance
app = FastAPI(
    title="Intent Classification API",
    description="An API to predict user intent and log feedback.",
    version="1.0.0"
)

# --- Pydantic Models for Request Body Validation ---

class PredictRequest(BaseModel):
    utterance: str

class FeedbackRequest(BaseModel):
    utterance: str
    predicted_intent: str
    is_correct: bool
    correct_intent: Optional[str] = None # This field is now optional

# --- API Endpoints ---

@app.post("/predict")
async def predict(request: PredictRequest):
    """Endpoint to get a prediction for a user's utterance."""
    predicted_intent = classifier.predict(request.utterance)
    
    return {
        "utterance": request.utterance,
        "predicted_intent": predicted_intent
    }

@app.post("/feedback", status_code=201)
async def feedback(request: FeedbackRequest):
    """Endpoint to log user feedback."""
    # Custom validation: if the prediction was wrong, user must provide the correct intent
    if not request.is_correct and not request.correct_intent:
        raise HTTPException(
            status_code=400, 
            detail="Field 'correct_intent' is required when 'is_correct' is false"
        )

    log_feedback(
        utterance=request.utterance,
        predicted_intent=request.predicted_intent,
        is_correct=request.is_correct,
        correct_intent=request.correct_intent
    )
    
    return {"status": "Feedback received"}