import os
import sys
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from langchain.chat_models import init_chat_model
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
from qdrant_client.models import VectorParams, Distance, CollectionConfig

# ==============================
# 1️⃣ Configuration
# ==============================
load_dotenv()

QDRANT_URL = "https://08ab67b6-8169-40a8-bfc6-96fb50f3743c.eu-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "URL_Data"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
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

# ==============================
# 2️⃣ Fetch & Split Documents
# ==============================
async def fetch_url_content(session, url, intent):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with session.get(url, headers=headers, timeout=10) as response:
            if response.status != 200:
                print(f"Failed to fetch {url} (Status: {response.status})")
                return None
            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")
            page_content = soup.get_text(separator=" ", strip=True)
            return Document(page_content=page_content, metadata={"source": url, "intent": intent})
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

async def load_and_split_docs():
    all_docs = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url_content(session, url, intent)
                 for intent, urls in INTENT_URL_MAP.items()
                 for url in urls]
        results = await asyncio.gather(*tasks)
        all_docs = [doc for doc in results if doc is not None]

    if not all_docs:
        print("❌ No documents loaded!")
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    print(f"✅ Split into {len(chunks)} document chunks.")
    return chunks

# ==============================
# 3️⃣ Ingest Documents to Qdrant
# ==============================
def ingest_to_qdrant(documents):
    if not documents:
        print("❌ No documents to ingest.")
        return

    print("🔹 Loading embedding model ...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("🔹 Connecting to Qdrant ...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

    print(f"🔹 Re-creating collection '{COLLECTION_NAME}' ...")
    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
            on_disk_payload=True
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="intent",
            field_type="keyword"
        )
        print(f"✅ Collection '{COLLECTION_NAME}' created and indexed successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not recreate collection: {e}")
        print("Attempting to proceed...")

    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embedding_model)
    print(f"🔹 Ingesting {len(documents)} documents ...")
    vector_store.add_documents(documents)
    print(f"✅ Successfully ingested {len(documents)} chunks.")

# ==============================
# 4️⃣ Initialize RAG Components
# ==============================
print("🔹 Loading embedding model for RAG ...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embedding_model)

print("🔹 Initializing Gemini LLM ...")
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0.3, google_api_key=GEMINI_API_KEY)

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant for an airline company.
Use only the provided context to answer the question.
If the context doesn't contain the answer, respond with:
"I'm sorry, I don't have that information in my documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""")

output_parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ==============================
# 5️⃣ RAG Query Function
# ==============================
async def get_rag_answer(query: str, human_intent: str) -> str:
    if human_intent not in HUMAN_INTENT_MAP:
        return "❌ Invalid intent. Please select a valid option."

    intent_key = HUMAN_INTENT_MAP[human_intent]
    print(f"\n🧠 Received query: '{query}' for intent: '{human_intent}' ({intent_key})")

    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3, "filter": {"intent": intent_key}}
        )

        retrieval_chain = RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
        rag_chain = retrieval_chain | prompt | llm | output_parser

        response = await rag_chain.ainvoke(query)
        print("✅ Generated response:", response)
        return response
    except Exception as e:
        print(f"❌ Error during RAG chain execution: {e}")
        return "I'm sorry, I encountered an error while generating an answer."

# ==============================
# 6️⃣ Main Async Ingestion
# ==============================
async def main():
    docs = await load_and_split_docs()
    ingest_to_qdrant(docs)
    print("\n🚀 Ingestion complete!")

if __name__ == "__main__":
    asyncio.run(main())