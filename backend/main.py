from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import your existing classifier and feedback logger
from app.predict import classifier
from app.feedback import log_feedback
from app.rag_service import get_rag_answer, is_rag_ready, get_available_intents

# Load environment variables
load_dotenv()

# Create the FastAPI app instance
app = FastAPI(
    title="Airline Chatbot API",
    description="An API for intent classification and RAG-based Q&A for airline customer service.",
    version="1.0.0"
)

# --- CORS Middleware ---
origins = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # React dev server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class PredictRequest(BaseModel):
    utterance: str = Field(..., min_length=1, description="User's input message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "utterance": "I want to cancel my flight"
            }
        }

class PredictResponse(BaseModel):
    utterance: str
    predicted_intent: str

class FeedbackRequest(BaseModel):
    utterance: str
    predicted_intent: str
    is_correct: bool
    correct_intent: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "utterance": "I want to cancel my flight",
                "predicted_intent": "cancel_trip",
                "is_correct": True
            }
        }

class RagRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's follow-up question")
    intent: str = Field(..., description="Previously classified intent")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the cancellation charges?",
                "intent": "cancel_trip"
            }
        }

class RagResponse(BaseModel):
    query: str
    intent: str
    answer: str

class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    rag_available: bool

# --- API Endpoints ---

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Airline Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "intents": "/intents",
            "predict": "/predict",
            "feedback": "/feedback",
            "rag_answer": "/rag-answer"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify system status."""
    return HealthResponse(
        status="healthy",
        classifier_loaded=classifier is not None,
        rag_available=is_rag_ready()
    )

@app.get("/intents")
async def get_intents():
    """Returns a list of all possible intent labels."""
    # Get labels from classifier model
    labels = list(classifier.model.config.id2label.values())
    return {
        "intents": sorted(labels),
        "count": len(labels)
    }

@app.get("/rag-intents")
async def get_rag_intents():
    """Returns a mapping of human-readable intents to database keys for RAG."""
    return {
        "intents": get_available_intents(),
        "count": len(get_available_intents())
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Endpoint to get intent prediction for user's utterance."""
    try:
        predicted_intent = classifier.predict(request.utterance)
        return PredictResponse(
            utterance=request.utterance,
            predicted_intent=predicted_intent
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )

@app.post("/feedback", status_code=201)
async def feedback(request: FeedbackRequest):
    """Endpoint to log user feedback for model improvement."""
    # Validation: if prediction was wrong, user must provide correct intent
    if not request.is_correct and not request.correct_intent:
        raise HTTPException(
            status_code=400, 
            detail="Field 'correct_intent' is required when 'is_correct' is false"
        )

    try:
        log_feedback(
            utterance=request.utterance,
            predicted_intent=request.predicted_intent,
            is_correct=request.is_correct,
            correct_intent=request.correct_intent
        )
        return {"status": "Feedback received", "message": "Thank you for your feedback!"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error logging feedback: {str(e)}"
        )

@app.post("/rag-answer", response_model=RagResponse)
async def rag_answer(request: RagRequest):
    """
    Endpoint to get RAG-generated answer based on user query and classified intent.
    
    This endpoint retrieves relevant documents from the knowledge base
    and generates a contextual answer using the Gemini LLM.
    """
    if not is_rag_ready():
        raise HTTPException(
            status_code=503,
            detail="RAG service is not available. Please check server logs."
        )
    
    print(f"📥 Received RAG request | Intent: {request.intent} | Query: {request.query}")
    
    try:
        answer = await get_rag_answer(request.query, request.intent)
        return RagResponse(
            query=request.query,
            intent=request.intent,
            answer=answer
        )
    except Exception as e:
        print(f"❌ Error in RAG endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    print("="*60)
    print("🚀 Starting Airline Chatbot API")
    print("="*60)
    print(f"✅ Classifier loaded: {classifier is not None}")
    print(f"✅ RAG service available: {is_rag_ready()}")
    print("="*60)

# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)