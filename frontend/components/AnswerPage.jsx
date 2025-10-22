import React from 'react';
import { useState, useRef, useEffect } from 'react';
import { API_BASE_URL } from '../src/App.jsx'; // Import the shared base URL

// We receive 'intent' and 'onBack' as props from App.jsx
function AnswerPage({ intent, onBack }) {
    const [query, setQuery] = useState('');
    const [messages, setMessages] = useState([]); // Stores the chat history
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null); // To auto-scroll chat
  
    // Add a welcome message when the component loads
    useEffect(() => {
      setMessages([
        { 
          sender: 'bot', 
          text: `Hi! I'm ready to answer questions about: ${intent}. What would you like to know?` 
        }
      ]);
    }, [intent]);
  
    // Auto-scroll to the bottom when new messages are added
    useEffect(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);
  
    const handleSubmit = async (e) => {
      e.preventDefault();
      if (!query.trim() || isLoading) return;
  
      const userMessage = { sender: 'user', text: query };
      setMessages(prev => [...prev, userMessage]);
      setIsLoading(true); // <-- FIX: Was 'True'
      setQuery('');
  
      try {
        // Call the new RAG endpoint
        const response = await fetch(`${API_BASE_URL}/rag-answer`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, intent }),
        });
  
        if (!response.ok) {
          // Try to get more error details from the backend
          const errData = await response.json();
          throw new Error(errData.detail || 'Failed to get an answer.');
        }
        
        const data = await response.json();
        const botMessage = { sender: 'bot', text: data.answer };
        setMessages(prev => [...prev, botMessage]);
  
      } catch (err) {
        const errorMessage = { sender: 'bot', text: `Sorry, an error occurred: ${err.message}` };
        setMessages(prev => [...prev, errorMessage]);
      } finally {
        setIsLoading(false); // <-- FIX: Was 'False'
      }
    };
  
    return (
      <>
        <button onClick={onBack} className="back-button">
          ← Back to Intent Prediction
        </button>
        
        <div className="chat-header">
          <h2>RAG Assistant</h2>
          <p>Current Topic: <code>{intent}</code></p>
        </div>
  
        <div className="chat-window">
          {messages.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.sender}`}>
              <p>{msg.text}</p>
            </div>
          ))}
          {isLoading && (
            <div className="chat-message bot">
              <p className="typing-indicator">...</p>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
  
        <form onSubmit={handleSubmit} className="input-form chat-input">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a follow-up question..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>
            Send
          </button>
        </form>
      </>
    );
  }
  
  export default AnswerPage;