from collections import deque
import joblib
import os

CACHE_FILE = "cache/recent_predictions.pkl"
CACHE_SIZE = 50

# Load cache
def load_cache():
    if os.path.exists(CACHE_FILE):
        return joblib.load(CACHE_FILE)
    return deque(maxlen=CACHE_SIZE)

# Save cache
def save_cache(cache_deque):
    joblib.dump(cache_deque, CACHE_FILE)

# Add new entry
def add_to_cache(utterance, correct_label):
    cache = load_cache()
    cache.append({"utterance": utterance, "correct_label": correct_label})
    save_cache(cache)
