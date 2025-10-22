import os
import sys
import asyncio
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ==============================
# 1️⃣ Configuration
# ==============================
load_dotenv()

QDRANT_URL = "https://08ab67b6-8169-40a8-bfc6-96fb50f3743c.eu-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
COLLECTION_NAME = "URL_Data"

if not QDRANT_API_KEY or not GEMINI_API_KEY:
    print("❌ Missing QDRANT_API_KEY or GOOGLE_API_KEY environment variable!", file=sys.stderr)
    sys.exit(1)

# ==============================
# 2️⃣ Load Global Components
# ==============================
try:
    print("🔹 Loading embedding model: all-MiniLM-L6-v2 ...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("🔹 Connecting to Qdrant Cloud ...")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        check_compatibility=False
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
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

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""")

    # Output parser
    output_parser = StrOutputParser()

except Exception as e:
    print(f"❌ FATAL: Could not initialize RAG service: {e}", file=sys.stderr)
    sys.exit(1)

# ==============================
# 3️⃣ RAG Handler
# ==============================
def format_docs(docs):
    """Combine retrieved docs into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

async def get_rag_answer(query: str, intent: str) -> str:
    if not intent:
        return "I'm sorry, I can't answer that without a valid intent. Please select an intent."

    print(f"\n🧠 Received query: '{query}' for intent: '{intent}'")

    try:
        # Proper Qdrant filter for string field
        filter_obj = Filter(
            must=[
                FieldCondition(
                    key="intent",
                    match=MatchValue(value=intent)
                )
            ]
        )

        # Run similarity search
        docs = await asyncio.to_thread(
            lambda: vector_store.similarity_search(
                query,
                k=3,
                filter=filter_obj
            )
        )

        if not docs:
            return "I'm sorry, I don't have that information in my documents."

        # Combine documents
        context = format_docs(docs)

        # Run LLM
        llm_input = {"context": context, "question": query}
        response = await llm.ainvoke(llm_input | prompt | output_parser)

        print("✅ Generated response:", response)
        return response

    except Exception as e:
        print(f"❌ Error during RAG chain execution: {e}", file=sys.stderr)
        return "I'm sorry, I encountered an error while generating an answer."