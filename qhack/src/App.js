import React from 'react';
import { HashRouter as Router, Route, Routes } from 'react-router-dom';
import BurnAssessmentApp from './landing';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<BurnAssessmentApp />} />
      </Routes>
    </Router>
  );
}

export default App;