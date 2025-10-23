"""
Fixed Ingestion Script for Airline Chatbot RAG
Properly structures metadata for Qdrant filtering
"""

import os
import sys
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from uuid import uuid4

# ==============================
# Configuration
# ==============================
load_dotenv()

QDRANT_URL = "https://08ab67b6-8169-40a8-bfc6-96fb50f3743c.eu-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "URL_Data"
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2

if not QDRANT_API_KEY:
    print("❌ Missing QDRANT_API_KEY environment variable!", file=sys.stderr)
    sys.exit(1)

# Map of intents -> URLs
INTENT_URL_MAP = {
    "cancel_trip": [
        "https://mybiz.makemytrip.com/corporate/flight-cancellation-charges.html",
        "https://www.airarabia.com/en/cancel-flight"
    ],
    "cancellation_policy": [
        "https://www.airindia.com/in/en/manage/request-refund.html",
        "https://www.transportation.gov/individuals/aviation-consumer-protection/refunds"
    ],
    "carry_on_luggage_faq": [
        "https://www.airindia.com/in/en/frequently-asked-questions/baggage.html"
    ],
    "change_flight": [
        "https://www.godigit.com/explore/flight-guide/how-to-reschedule-flight"
    ],
    "check_in_luggage_faq": [
        "https://www.britishairways.com/content/information/baggage-essentials"
    ],
    "complaints": [
        "https://www.civilaviation.gov.in/vigilance/how-register-complaint"
    ],
    "damaged_bag": [
        "https://www.transportation.gov/lost-delayed-or-damaged-baggage"
    ],
    "discounts": [
        "https://www.cleartrip.com/flights"
    ],
    "fare_check": [
        "https://www.altexsoft.com/glossary/fare-rules/"
    ],
    "flight_status": [
        "https://www.ixigo.com/flight-status"
    ],
    "flights_info": [
        "https://www.newdelhiairport.in/live-flight-information/"
    ],
    "insurance": [
        "https://www.tataaig.com/travel-insurance"
    ],
    "medical_policy": [
        "https://www.airindia.com/in/en/travel-information/health-medical-assistance/medical-needs-clearance.html"
    ],
    "missing_bag": [
        "https://www.goindigo.in/add-on-services/delayed-and-lost-baggage-protection.html"
    ],
    "pet_travel": [
        "https://www.airindia.com/in/en/travel-information/first-time-flyers/carriage-of-pets.html",
        "https://www.jetblue.com/traveling-together/traveling-with-pets"
    ],
    "prohibited_items_faq": [
        "https://www.emirates.com/in/english/before-you-fly/travel/dangerous-goods-policy/"
    ],
    "seat_availability": [
        "https://www.air.irctc.co.in/air-services/flight-tickets-seat-availability.html"
    ],
    "sports_music_gear": [
        "https://support.fly91.in/portal/en/kb/articles/sports-equipment-special-baggage"
    ]
}

# ==============================
# Fetch & Split Documents
# ==============================
async def fetch_url_content(session, url, intent):
    """Fetch content from URL and create Document."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with session.get(url, headers=headers, timeout=15) as response:
            if response.status != 200:
                print(f"⚠️  Failed to fetch {url} (Status: {response.status})")
                return None
            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            
            page_content = soup.get_text(separator=" ", strip=True)
            
            # Clean up extra whitespace
            page_content = " ".join(page_content.split())
            
            if len(page_content) < 100:
                print(f"⚠️  Skipping {url} - insufficient content")
                return None
            
            print(f"✅ Fetched {url} ({len(page_content)} chars)")
            return Document(
                page_content=page_content, 
                metadata={"source": url, "intent": intent}
            )
    except asyncio.TimeoutError:
        print(f"⚠️  Timeout fetching {url}")
        return None
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return None

async def load_and_split_docs():
    """Load and split all documents."""
    print("\n" + "="*70)
    print("FETCHING DOCUMENTS")
    print("="*70 + "\n")
    
    all_docs = []
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_url_content(session, url, intent)
            for intent, urls in INTENT_URL_MAP.items()
            for url in urls
        ]
        results = await asyncio.gather(*tasks)
        all_docs = [doc for doc in results if doc is not None]

    if not all_docs:
        print("❌ No documents loaded!")
        return []

    print(f"\n✅ Loaded {len(all_docs)} documents")
    print("\n" + "="*70)
    print("SPLITTING DOCUMENTS")
    print("="*70 + "\n")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)
    
    print(f"✅ Split into {len(chunks)} chunks")
    
    # Print summary by intent
    from collections import Counter
    intent_counts = Counter(chunk.metadata["intent"] for chunk in chunks)
    print("\n📊 Chunks per intent:")
    for intent, count in sorted(intent_counts.items()):
        print(f"   {intent}: {count} chunks")
    
    return chunks

# ==============================
# Direct Qdrant Ingestion (bypassing LangChain)
# ==============================
def ingest_to_qdrant(documents):
    """
    Ingest documents directly to Qdrant with proper metadata structure.
    This bypasses LangChain's wrapper to ensure metadata is stored correctly.
    """
    if not documents:
        print("❌ No documents to ingest.")
        return

    print("\n" + "="*70)
    print("INITIALIZING QDRANT")
    print("="*70 + "\n")

    print("🔹 Loading embedding model: all-MiniLM-L6-v2 ...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("🔹 Connecting to Qdrant Cloud ...")
    client = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY, 
        check_compatibility=False
    )

    print(f"🔹 Recreating collection '{COLLECTION_NAME}' ...")
    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
        )
        print(f"✅ Collection '{COLLECTION_NAME}' created successfully.")
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return

    # Create payload index for intent field (at root level)
    print("🔹 Creating payload index for 'intent' field ...")
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="intent",
            field_schema="keyword"
        )
        print("✅ Payload index created successfully.")
    except Exception as e:
        print(f"⚠️  Warning: Could not create payload index: {e}")

    print("\n" + "="*70)
    print("INGESTING DOCUMENTS")
    print("="*70 + "\n")

    # Batch processing
    batch_size = 100
    total_points = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        points = []
        
        print(f"🔹 Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} docs)...")
        
        # Generate embeddings for batch
        texts = [doc.page_content for doc in batch]
        embeddings = embedding_model.embed_documents(texts)
        
        # Create points with metadata at ROOT LEVEL (not nested)
        for doc, embedding in zip(batch, embeddings):
            point = PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "page_content": doc.page_content,
                    "intent": doc.metadata["intent"],  # At root level
                    "source": doc.metadata["source"],  # At root level
                }
            )
            points.append(point)
        
        # Upload batch
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            total_points += len(points)
            print(f"   ✅ Uploaded {len(points)} points")
        except Exception as e:
            print(f"   ❌ Error uploading batch: {e}")
    
    print(f"\n✅ Successfully ingested {total_points} chunks to Qdrant!")
    
    # Verify ingestion
    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"✅ Collection now has {collection_info.points_count} points")

# ==============================
# Main
# ==============================
async def main():
    print("\n" + "="*70)
    print("AIRLINE CHATBOT - DATA INGESTION")
    print("="*70)
    
    docs = await load_and_split_docs()
    
    if docs:
        ingest_to_qdrant(docs)
        print("\n" + "="*70)
        print("✅ INGESTION COMPLETE!")
        print("="*70 + "\n")
    else:
        print("\n❌ Ingestion failed - no documents to process")

if __name__ == "__main__":
    asyncio.run(main())