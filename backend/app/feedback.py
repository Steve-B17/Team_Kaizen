import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from typing import Optional

# --- Setup ---
load_dotenv()

# --- Constants ---
DB_SECRET_NAME = "POSTGRES_CONNECTION_STRING"

# --- Database Utility ---
def get_db_engine():
    """Establishes a connection to the PostgreSQL database."""
    connection_string = os.environ.get(DB_SECRET_NAME)
    if not connection_string:
        print(f"❌ Set {DB_SECRET_NAME} in your .env file.", file=sys.stderr)
        sys.exit(1)
    try:
        # We don't need to print the success message every time.
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}", file=sys.stderr)
        sys.exit(1)

def log_feedback(
    utterance: str, 
    predicted_intent: str, 
    is_correct: bool, 
    correct_intent: Optional[str] = None
):
    """
    Logs the user's feedback into the 'feedback_logs' table in the database.
    """
    engine = get_db_engine()
    
    # SQL query to insert feedback data.
    # We also add a 'created_at' timestamp automatically.
    query = text(
        """
        INSERT INTO feedback_logs (
            utterance, 
            predicted_intent, 
            is_correct, 
            correct_intent,
            created_at
        )
        VALUES (
            :utterance, 
            :predicted_intent, 
            :is_correct, 
            :correct_intent,
            NOW()
        )
        """
    )
    
    try:
        with engine.connect() as connection:
            connection.execute(query, {
                "utterance": utterance,
                "predicted_intent": predicted_intent,
                "is_correct": is_correct,
                "correct_intent": correct_intent
            })
            connection.commit() # Make sure to commit the transaction
        print("✅ Feedback logged successfully.")
    except Exception as e:
        print(f"❌ Error logging feedback: {e}")
