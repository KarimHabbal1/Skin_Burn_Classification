import React from 'react';

function ResultSection({ isAnalyzing, result }) {
  if (isAnalyzing) {
    return (
      <div className="result-section analyzing">
        <div className="loader"></div>
        <p>Analyzing with quantum-assisted algorithm...</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="result-section empty">
        <div className="placeholder-content">
          <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
          <h3>Assessment Results</h3>
          <p>Upload and analyze an image to see AI assessment results here.</p>
        </div>
      </div>
    );
  }

  // Determine color based on burn degree
  let severityColor = "#28a745"; // green for first degree
  if (result.degree === "Second Degree") {
    severityColor = "#ffc107"; // yellow for second degree
  } else if (result.degree === "Third Degree") {
    severityColor = "#dc3545"; // red for third degree
  }

  return (
    <div className="result-section">
      <h3>Assessment Results</h3>
      
      <div className="result-card">
        <div className="result-header" style={{ backgroundColor: severityColor }}>
          <h4>{result.degree} Burn</h4>
          <div className="confidence-badge">
            {result.confidence}% confidence
          </div>
        </div>
        
        <div className="result-body">
          <h5>Recommendations</h5>
          <p>{result.recommendations}</p>
          
          <div className="disclaimer-box">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" y1="8" x2="12" y2="12"/>
              <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            <p>This is not a medical diagnosis. Always seek professional medical advice for burn treatment.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ResultSection;