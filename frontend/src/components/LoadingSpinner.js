import React from 'react';
import './LoadingSpinner.css';

const LoadingSpinner = () => {
  return (
    <div className="loading-state">
      <div className="quantum-loader">
        <div className="loader-circle"></div>
        <p>Training quantum model...</p>
      </div>
    </div>
  );
};

export default LoadingSpinner;
