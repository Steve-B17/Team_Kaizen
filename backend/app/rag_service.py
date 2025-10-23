"""
Fixed RAG Service Module for Airline Chatbot
Works with properly structured Qdrant metadata
"""

import os
import sys
import asyncio
from typing import Dict, Optional, List
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# ==============================
# Configuration
# ==============================
load_dotenv()

QDRANT_URL = "https://08ab67b6-8169-40a8-bfc6-96fb50f3743c.eu-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
COLLECTION_NAME = "URL_Data"

if not QDRANT_API_KEY or not GEMINI_API_KEY:
    print("❌ Missing QDRANT_API_KEY or GOOGLE_API_KEY environment variable!", file=sys.stderr)
    print("⚠️ RAG service will not be available.", file=sys.stderr)

# Map human-readable intent names to database keys
HUMAN_INTENT_MAP = {
    "Cancel Trip": "cancel_trip",
    "Cancellation Policy": "cancellation_policy",
    "Carry-On Luggage FAQ": "carry_on_luggage_faq",
    "Change Flight": "change_flight",
    "Check-in Luggage FAQ": "check_in_luggage_faq",
    "Complaints": "complaints",
    "Damaged Bag": "damaged_bag",
    "Discounts": "discounts",
    "Fare Check": "fare_check",
    "Flight Status": "flight_status",
    "Flights Info": "flights_info",
    "Insurance": "insurance",
    "Medical Policy": "medical_policy",
    "Missing Bag": "missing_bag",
    "Pet Travel": "pet_travel",
    "Prohibited Items FAQ": "prohibited_items_faq",
    "Seat Availability": "seat_availability",
    "Sports Music Gear": "sports_music_gear"
}

# Also support lowercase snake_case intents directly from classifier
INTENT_KEY_MAP = {v: v for v in HUMAN_INTENT_MAP.values()}

# Related intents that should be searched together
RELATED_INTENTS = {
    "flights_info": ["flights_info", "flight_status"],
    "flight_status": ["flights_info", "flight_status"],
}

# ==============================
# Global Components
# ==============================
embedding_model = None
client = None
llm = None
prompt = None
output_parser = None
rag_initialized = False

# ==============================
# Initialization
# ==============================
def initialize_rag():
    """Initialize RAG components. Called once on module load."""
    global embedding_model, client, llm, prompt, output_parser, rag_initialized
    
    if rag_initialized:
        print("✅ RAG already initialized, skipping...")
        return True
    
    try:
        print("🔹 Loading embedding model: all-MiniLM-L6-v2 ...")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        print("🔹 Connecting to Qdrant Cloud ...")
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            check_compatibility=False
        )
        print("✅ Connected to Qdrant successfully.")

        print("🔹 Initializing Gemini LLM ...")
        llm = init_chat_model(
            "gemini-2.5-flash",
            model_provider="google_genai",
            temperature=0.3,
            google_api_key=GEMINI_API_KEY
        )
        print("✅ Gemini LLM loaded successfully.")

        # Prompt template
        prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant for an airline company.
Use only the provided context to answer the question.
If the context doesn't contain the answer, respond with:
"I'm sorry, I don't have that information in my documents."

Keep your answers concise and relevant to the question.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""")

        output_parser = StrOutputParser()
        
        rag_initialized = True
        print("✅ RAG system initialized successfully!\n")
        return True

    except Exception as e:
        print(f"❌ Could not initialize RAG service: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        rag_initialized = False
        return False

# Initialize on module import
initialize_rag()

# ==============================
# Helper Functions
# ==============================
def normalize_intent(intent: str) -> Optional[str]:
    """
    Convert any intent format to the database key format.
    Handles: "Cancel Trip", "cancel_trip", "Cancel trip", etc.
    
    Returns the database key or None if invalid.
    """
    # First check if it's already a valid database key
    if intent in INTENT_KEY_MAP:
        return intent
    
    # Check if it's a human-readable format
    if intent in HUMAN_INTENT_MAP:
        return HUMAN_INTENT_MAP[intent]
    
    # Try case-insensitive matching for human-readable
    for human_intent, db_key in HUMAN_INTENT_MAP.items():
        if human_intent.lower() == intent.lower():
            return db_key
    
    # Try converting to snake_case and matching
    intent_snake = intent.lower().replace(" ", "_")
    if intent_snake in INTENT_KEY_MAP:
        return intent_snake
    
    return None

def search_with_filter(query_embedding: List[float], intents: List[str], limit: int = 5):
    """
    Search Qdrant with intent filter using native client.
    Metadata is stored at ROOT level (not nested).
    """
    # Create filter for intent(s)
    if len(intents) == 1:
        filter_obj = Filter(
            must=[
                FieldCondition(
                    key="intent",  # Direct field at root level
                    match=MatchValue(value=intents[0])
                )
            ]
        )
    else:
        filter_obj = Filter(
            must=[
                FieldCondition(
                    key="intent",  # Direct field at root level
                    match=MatchAny(any=intents)
                )
            ]
        )
    
    # Search using native Qdrant client
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        query_filter=filter_obj,
        limit=limit,
        with_payload=True
    )
    
    return search_result

# ==============================
# Main RAG Function
# ==============================
async def get_rag_answer(query: str, intent: str) -> str:
    """
    Retrieve relevant documents and generate an answer using RAG.
    
    Args:
        query: User's question (e.g., "What are the cancellation charges?")
        intent: Intent from classifier (e.g., "cancel_trip" or "Cancel Trip")
    
    Returns:
        Generated answer string
    """
    if not rag_initialized:
        return "❌ RAG system is not available. Please check the server logs."
    
    # Normalize the intent to database key format
    intent_key = normalize_intent(intent)
    
    if not intent_key:
        print(f"⚠️ Invalid intent received: '{intent}'")
        return f"❌ Invalid intent '{intent}'. Please use a valid intent category."
    
    print(f"\n🧠 Processing query: '{query}' | Intent: '{intent}' → '{intent_key}'")

    try:
        # Check if this intent should search multiple related intents
        intents_to_search = RELATED_INTENTS.get(intent_key, [intent_key])
        print(f"🔍 Searching intents: {intents_to_search}")
        
        # Generate query embedding
        query_embedding = await asyncio.to_thread(
            lambda: embedding_model.embed_query(query)
        )
        
        # Search with intent filter
        search_results = await asyncio.to_thread(
            lambda: search_with_filter(query_embedding, intents_to_search, limit=5)
        )
        
        if not search_results:
            print("⚠️ No documents found with intent filter. Trying broader search...")
            # Fallback: search without filter
            search_results = await asyncio.to_thread(
                lambda: client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_embedding,
                    limit=3,
                    with_payload=True
                )
            )
            if search_results:
                print(f"✅ Found {len(search_results)} documents without intent filter")
        
        if not search_results:
            print("⚠️ No relevant documents found.")
            return "I'm sorry, I don't have specific information about that in my documents. Please try rephrasing your question or contact customer support for more details."

        print(f"📄 Retrieved {len(search_results)} relevant documents")
        
        # Extract page_content from search results
        contexts = []
        for result in search_results:
            content = result.payload.get("page_content", "")
            if content:
                contexts.append(content)
            print(f"   📌 Intent: {result.payload.get('intent')} | Score: {result.score:.3f}")
        
        if not contexts:
            return "I'm sorry, I couldn't extract relevant information from the documents."
        
        # Combine contexts
        combined_context = "\n\n".join(contexts)
        
        # Create the chain and invoke it
        chain = prompt | llm | output_parser
        response = await chain.ainvoke({
            "context": combined_context, 
            "question": query
        })

        print("✅ Generated response successfully")
        return response

    except Exception as e:
        print(f"❌ Error during RAG chain execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return "I'm sorry, I encountered an error while generating an answer. Please try again or contact support."

# ==============================
# Utility Functions
# ==============================
def get_available_intents() -> Dict[str, str]:
    """Returns mapping of human-readable intents to database keys."""
    return HUMAN_INTENT_MAP.copy()

def is_rag_ready() -> bool:
    """Check if RAG system is initialized and ready."""
    return rag_initialized

async def debug_search(query: str, intent: Optional[str] = None):
    """
    Debug function to inspect search results and metadata structure.
    
    Args:
        query: Search query
        intent: Optional intent to filter by
    """
    if not rag_initialized:
        print("❌ RAG not initialized")
        return
    
    print(f"\n{'='*70}")
    print(f"🔍 Debug search for: '{query}'")
    if intent:
        print(f"   Intent filter: '{intent}'")
    print('='*70)
    
    # Generate embedding
    query_embedding = await asyncio.to_thread(
        lambda: embedding_model.embed_query(query)
    )
    
    # Search without filter
    results_no_filter = await asyncio.to_thread(
        lambda: client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=5,
            with_payload=True
        )
    )
    
    print(f"\n📊 Results WITHOUT filter: {len(results_no_filter)} documents")
    for i, result in enumerate(results_no_filter, 1):
        print(f"\n--- Document {i} (Score: {result.score:.3f}) ---")
        print(f"Intent: {result.payload.get('intent')}")
        print(f"Source: {result.payload.get('source')}")
        print(f"Payload keys: {list(result.payload.keys())}")
        content = result.payload.get('page_content', '')
        print(f"Content preview: {content[:200]}...")
    
    # Search with filter if intent provided
    if intent:
        intent_key = normalize_intent(intent)
        if intent_key:
            intents_to_search = RELATED_INTENTS.get(intent_key, [intent_key])
            
            print(f"\n{'='*70}")
            print(f"🔍 Searching with filter: intent in {intents_to_search}")
            print('='*70)
            
            results_with_filter = await asyncio.to_thread(
                lambda: search_with_filter(query_embedding, intents_to_search, limit=5)
            )
            
            print(f"\n📊 Results WITH filter: {len(results_with_filter)} documents")
            for i, result in enumerate(results_with_filter, 1):
                print(f"\n--- Document {i} (Score: {result.score:.3f}) ---")
                print(f"Intent: {result.payload.get('intent')}")
                print(f"Content preview: {result.payload.get('page_content', '')[:200]}...")
    
    print(f"\n{'='*70}\n")
    return results_no_filter

# ==============================
# Test Function
# ==============================
async def test_rag():
    """Test function to verify RAG is working."""
    if not rag_initialized:
        print("❌ Cannot test - RAG not initialized")
        return
    
    test_cases = [
        ("coimbatore to hyderabad flights", "flight_status"),
        ("What are the cancellation charges?", "cancel_trip"),
        ("Can I bring my dog on the flight?", "pet_travel"),
        ("What items cannot be carried?", "prohibited_items_faq"),
    ]
    
    print("\n" + "="*70)
    print("RAG SYSTEM TESTS")
    print("="*70)
    
    for query, intent in test_cases:
        print(f"\n{'='*70}")
        print(f"Test: {query} | Intent: {intent}")
        print('='*70)
        answer = await get_rag_answer(query, intent)
        print(f"\n💬 Answer: {answer}\n")
        print("-"*70)

if __name__ == "__main__":
    # Run tests if executed directly
    asyncio.run(test_rag())