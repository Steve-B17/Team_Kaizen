import React from 'react';
import { useState, useEffect } from 'react';
import { API_BASE_URL } from "../src/App.jsx";

// We pass in 'onPredictionCorrect' as a prop from App.jsx
function PredictionPage({ onPredictionCorrect }) {
  // State for the user's input
  const [utterance, setUtterance] = useState('');
  
  // State for the API prediction result
  const [prediction, setPrediction] = useState(null);
  
  // State to handle loading and errors
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // State for the feedback mechanism
  const [showFeedbackOptions, setShowFeedbackOptions] = useState(false);
  const [selectedIntent, setSelectedIntent] = useState('');
  const [allIntents, setAllIntents] = useState([]);

  // Fetch all possible intents when the component loads
  useEffect(() => {
    const fetchIntents = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/intents`);
        if (!response.ok) throw new Error('Failed to fetch intents.');
        const data = await response.json();
        
        // ✅ FIX: The API returns {intents: [...], count: N}
        // So we need to access data.intents, not just data
        setAllIntents(data.intents || []);
      } catch (err) {
        console.error('Error fetching intents:', err);
        setError('Could not connect to API to get intents.');
      }
    };
    fetchIntents();
  }, []);

  // --- Handlers ---

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!utterance.trim()) return;

    setIsLoading(true);
    setError('');
    setPrediction(null);
    setShowFeedbackOptions(false);
    setSelectedIntent(''); // ✅ Reset selected intent on new prediction

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ utterance }),
      });

      if (!response.ok) throw new Error('Prediction request failed.');
      
      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeedback = async (isCorrect) => {
    if (!prediction) return;

    // ✅ Validation: if incorrect, must select an intent
    if (!isCorrect && !selectedIntent) {
      alert('Please select the correct intent from the dropdown.');
      return;
    }

    let feedbackBody = {
      utterance: prediction.utterance,
      predicted_intent: prediction.predicted_intent,
      is_correct: isCorrect,
      correct_intent: isCorrect ? null : selectedIntent,
    };

    try {
      const response = await fetch(`${API_BASE_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feedbackBody),
      });

      if (!response.ok) {
        throw new Error('Failed to submit feedback');
      }

      if (isCorrect) {
        // Call the function from App.jsx to navigate to RAG page
        onPredictionCorrect(prediction.predicted_intent);
      } else {
        // Reset the UI after incorrect feedback is sent
        alert('Thank you for your feedback! We\'ll use this to improve the model.');
        setPrediction(null);
        setUtterance('');
        setShowFeedbackOptions(false);
        setSelectedIntent(''); // ✅ Reset selected intent
      }

    } catch (err) {
      setError('Failed to submit feedback: ' + err.message);
    }
  };

  // --- Render ---

  return (
    <>
      <h1>Intent Classifier App</h1>
      <p className="subtitle">Enter a sentence to predict its intent.</p>

      <form onSubmit={handlePredict} className="input-form">
        <input
          type="text"
          value={utterance}
          onChange={(e) => setUtterance(e.target.value)}
          placeholder="e.g., How much money is in my account?"
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Predicting...' : 'Predict'}
        </button>
      </form>

      {error && <p className="error-message">{error}</p>}

      {prediction && (
        <div className="result-card">
          <h3>Prediction Result</h3>
          <p><strong>Utterance:</strong> "{prediction.utterance}"</p>
          <p><strong>Predicted Intent:</strong> <code>{prediction.predicted_intent}</code></p>
          
          <hr />

          <div className="feedback-section">
            <p>Was this prediction correct?</p>
            {!showFeedbackOptions ? (
              <div className="button-group">
                <button className="btn-yes" onClick={() => handleFeedback(true)}>
                  ✅ Yes
                </button>
                <button className="btn-no" onClick={() => setShowFeedbackOptions(true)}>
                  ❌ No
                </button>
              </div>
            ) : (
              <div className="correction-form">
                <p style={{ color: '#666', fontSize: '14px', marginBottom: '8px' }}>
                  Please select the correct intent:
                </p>
                <select
                  value={selectedIntent}
                  onChange={(e) => setSelectedIntent(e.target.value)}
                >
                  <option value="" disabled>Select the correct intent...</option>
                  {allIntents.map(intent => (
                    <option key={intent} value={intent}>{intent}</option>
                  ))}
                </select>
                <div className="button-group">
                  <button 
                    onClick={() => handleFeedback(false)}
                    disabled={!selectedIntent}
                    style={{ opacity: selectedIntent ? 1 : 0.5 }}
                  >
                    Submit Correction
                  </button>
                  <button 
                    onClick={() => {
                      setShowFeedbackOptions(false);
                      setSelectedIntent('');
                    }}
                    className="btn-cancel"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}

export default PredictionPage;