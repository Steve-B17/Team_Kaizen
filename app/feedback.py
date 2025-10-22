# In app/feedback.py
from sqlalchemy import create_engine, text
import os

def get_db_engine():
    # Assumes you have the connection string in your .env file
    return create_engine(os.environ["POSTGRES_CONNECTION_STRING"])

def log_feedback(utterance: str, predicted_intent: str, is_correct: bool, correct_intent: str = None):
    engine = get_db_engine()
    query = text(
        """
        INSERT INTO feedback_logs (utterance, predicted_intent, is_correct, correct_intent)
        VALUES (:utterance, :predicted_intent, :is_correct, :correct_intent)
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