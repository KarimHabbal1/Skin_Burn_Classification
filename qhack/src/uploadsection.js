import React from 'react';

function UploadSection({ 
  previewUrl, 
  handleImageChange, 
  handleDrop, 
  handleDragOver, 
  analyzeImage, 
  resetForm, 
  fileInputRef,
  selectedImage 
}) {
  return (
    <div className="upload-section">
      <h3>Upload Burn Image</h3>
      
      <div 
        className="drop-area" 
        onDrop={handleDrop} 
        onDragOver={handleDragOver}
      >
        {previewUrl ? (
          <div className="preview-container">
            <img src={previewUrl} alt="Burn preview" className="image-preview" />
            <button className="btn-reset" onClick={resetForm}>
              Choose Different Image
            </button>
          </div>
        ) : (
          <>
            <div className="upload-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
            </div>
            <p>Drag & drop your image here or</p>
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleImageChange} 
              id="file-input" 
              ref={fileInputRef}
              className="hidden-input"
            />
            <label htmlFor="file-input" className="btn-upload">Select File</label>
          </>
        )}
      </div>
      
      <button 
        className={`btn-analyze ${!selectedImage ? 'disabled' : ''}`} 
        onClick={analyzeImage}
        disabled={!selectedImage}
      >
        Analyze Image
      </button>
    </div>
  );
}

export default UploadSection;