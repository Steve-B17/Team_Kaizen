from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware # <-- IMPORT THIS

# Import your existing classifier and feedback logger
from app.predict import classifier
from app.feedback import log_feedback
from app.rag_service import get_rag_answer # <-- ADD THIS IMPORT
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create the FastAPI app instance
app = FastAPI(
    title="Intent Classification API",
    description="An API to predict user intent and log feedback.",
    version="1.0.0"
)

# --- ADD THIS CORS MIDDLEWARE SECTION ---
origins = [
    "http://localhost:5173", # The default Vite dev server port
    "http://localhost:3000", # A common port for create-react-app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- Pydantic Models for Request Body Validation ---

class PredictRequest(BaseModel):
    utterance: str

class FeedbackRequest(BaseModel):
    utterance: str
    predicted_intent: str
    is_correct: bool
    correct_intent: Optional[str] = None # This field is now optional
    
class RagRequest(BaseModel):
    query: str   # The user's follow-up question (e.g., "what's the fee?")
    intent: str  # The intent classified by the first model (e.g., "pet_travel")

# --- API Endpoints ---

@app.get("/intents")
async def get_intents():
    """Returns a list of all possible intent labels."""
    # The labels are stored in the model's config
    labels = list(classifier.model.config.id2label.values())
    return sorted(labels)

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

@app.post("/rag-answer")
async def rag_answer(request: RagRequest):
    """
    Endpoint to get a RAG-generated answer based on a
    user query and a pre-classified intent.
    """
    print(f"Received RAG request for intent: {request.intent}")
    answer = await get_rag_answer(request.query, request.intent)
    return {
        "query": request.query,
        "intent": request.intent,
        "answer": answer
    }