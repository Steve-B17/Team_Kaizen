import React from 'react';
import{ useState } from 'react';
import PredictionPage from '../components/PredictionPage.jsx';
import AnswerPage from '../components/AnswerPage.jsx';

// The base URL of your FastAPI backend
export const API_BASE_URL = 'http://127.0.0.1:8000';

function App() {
  // 'currentView' controls which "page" is visible
  const [currentView, setCurrentView] = useState('predict'); // 'predict' or 'answer'
  
  // 'currentIntent' stores the intent to pass to the AnswerPage
  const [currentIntent, setCurrentIntent] = useState('');

  // This function is passed to PredictionPage to allow it to "redirect"
  const navigateToAnswers = (intent) => {
    setCurrentIntent(intent);
    setCurrentView('answer');
  };

  // This function is passed to AnswerPage to allow it to go back
  const navigateToPredict = () => {
    setCurrentIntent('');
    setCurrentView('predict');
  };

  // Render the correct page based on the current view state
  // IMPORTANT: Remove the container div wrapper
  return (
    <>
      {currentView === 'predict' && (
        <PredictionPage onPredictionCorrect={navigateToAnswers} API_BASE_URL={API_BASE_URL} />
      )}
      {currentView === 'answer' && (
        <AnswerPage 
          intent={currentIntent} 
          onBack={navigateToPredict} 
        />
      )}
    </>
  );
}

export default App;