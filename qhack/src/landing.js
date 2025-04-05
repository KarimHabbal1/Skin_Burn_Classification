// App.jsx
import React, { useState, useRef } from 'react';
import './landing.css';
import Header from './header';
import UploadSection from './uploadsection';
import ResultSection from './resultsection';
import Footer from './footer';

// Import the hero image
import heroImage from './assets/hero-image.png'; // Your medical professional image
// No need to import fire.png as we'll use it as a background-image in CSS

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const fileInputRef = useRef(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const analyzeImage = () => {
    if (!selectedImage) return;
    
    setIsAnalyzing(true);
    
    // Simulating API call to quantum-assisted ML model
    setTimeout(() => {
      // Mock result - in a real app, this would come from your backend
      const burnDegrees = ["First Degree", "Second Degree", "Third Degree"];
      const randomIndex = Math.floor(Math.random() * burnDegrees.length);
      
      setResult({
        degree: burnDegrees[randomIndex],
        confidence: Math.floor(Math.random() * 20 + 80), // 80-99% confidence
        recommendations: getRecommendations(burnDegrees[randomIndex])
      });
      
      setIsAnalyzing(false);
    }, 2000);
  };

  const getRecommendations = (degree) => {
    switch(degree) {
      case "First Degree":
        return "Apply cool water for 10-15 minutes. Use aloe vera or moisturizer. Take pain relievers if needed. No medical attention required unless it doesn't improve in a few days.";
      case "Second Degree":
        return "Run cool water over the burn. Don't break blisters. Apply antibiotic ointment. Seek medical attention if the burn is larger than 3 inches or on a sensitive area.";
      case "Third Degree":
        return "URGENT: Seek immediate medical attention. Do not remove clothing stuck to the burn. Do not apply ointments. Cover with clean cloth until medical help arrives.";
      default:
        return "Please consult a healthcare professional for proper assessment and treatment.";
    }
  };

  const resetForm = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="app">
      <Header />
      
      <div className="hero-section">
        <div className="hero-content">
          <h1>Burn Assessment AI</h1>
          <p>Quantum-assisted medical solutions for under-resourced regions and post-war recovery efforts</p>
        </div>
       
        
        {/* Fire effect at the bottom */}
        <div className="fire-container"></div>
      </div>
      
      <main className="main-container" id="assess-now">
        <section className="intro-section">
          <h2>Quantum-Powered Burn Analysis</h2>
          <p>Upload an image of a burn to get an instant AI-powered assessment. Our quantum machine learning model can help determine the burn degree and provide initial care recommendations.</p>
          <div className="disclaimer">
            <strong>Important:</strong> This tool is for informational purposes only and does not replace professional medical advice. Always consult healthcare professionals for proper diagnosis and treatment.
          </div>
        </section>

        <div className="content-container">
          <UploadSection 
            previewUrl={previewUrl}
            handleImageChange={handleImageChange}
            handleDrop={handleDrop}
            handleDragOver={handleDragOver}
            analyzeImage={analyzeImage}
            resetForm={resetForm}
            fileInputRef={fileInputRef}
            selectedImage={selectedImage}
          />

          <ResultSection 
            isAnalyzing={isAnalyzing} 
            result={result}
          />
        </div>
      </main>

      <Footer />
    </div>
  );
}

export default App;