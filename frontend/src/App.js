import React, { useState } from 'react';
import './App.css';

// --- Sub-components for a cleaner structure ---

const AnalysisSummary = ({ sentiment, summary }) => (
  <div className="card summary-card">
    <h2>Analysis Summary</h2>
    <div className="sentiment-badge" data-sentiment={sentiment.label.replace(' ', '-').toLowerCase()}>
      {sentiment.label}
      <span className="sentiment-score">Score: {sentiment.score.toFixed(2)}</span>
    </div>
    <p className="summary-text">{summary}</p>
  </div>
);

const EmotionCard = ({ emotion, isPrimary }) => (
  <div className="card emotion-card">
    <div className="emotion-header">
      <span className="emotion-emoji">{emotion.emoji}</span>
      <div className="emotion-title">
        <h3>{emotion.label}</h3>
        {isPrimary && <span className="primary-badge">Primary Emotion</span>}
      </div>
      <span className="emotion-score">{(emotion.score * 100).toFixed(0)}%</span>
    </div>
    <p className="emotion-explanation">{emotion.explanation}</p>
  </div>
);


// --- Main App Component ---

function App() {
  const [text, setText] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze.');
      setAnalysis(null);
      return;
    }

    setIsLoading(true);
    setError('');
    setAnalysis(null);

    try {
      const response = await fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || 'An unknown error occurred.');
      }
      
      const data = await response.json();
      setAnalysis(data);

    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleAnalyze();
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Emotion Detection</h1>
        <p>Discover the emotions hidden in your text with advanced AI analysis</p>
      </header>

      <main>
        <div className="card input-card">
          <div className="ai-badge">✨ AI-Powered Analysis</div>
          <h2>What's on your mind?</h2>
          <p className="input-subtitle">Share your thoughts, and I'll analyze the emotions within your text.</p>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type or paste your text here... Express yourself freely and let AI understand the emotions behind your words."
          />
          <div className="input-footer">
            <span className="char-counter">{text.length} characters</span>
            <button onClick={handleAnalyze} disabled={isLoading}>
              {isLoading ? (
                <div className="spinner"></div>
              ) : (
                'Analyze Emotions'
              )}
            </button>
          </div>
        </div>

        {error && <div className="card error-card">{error}</div>}

        {analysis && (
          <div className="results-section">
            <AnalysisSummary sentiment={analysis.sentiment} summary={analysis.summary} />
            <h2>Detected Emotions</h2>
            <div className="emotions-grid">
              {analysis.emotions.map((emotion, index) => (
                <EmotionCard key={emotion.label} emotion={emotion} isPrimary={index === 0} />
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;