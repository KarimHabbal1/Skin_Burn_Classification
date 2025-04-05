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

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const resetForm = () => {
    setPreviewUrl(null);
    setSelectedImage(null);
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error analyzing image:', error);
      setResult({
        degree: "Error",
        confidence: 0,
        recommendations: "An error occurred while analyzing the image. Please try again."
      });
    } finally {
      setIsAnalyzing(false);
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