import React from 'react';
import './Home.css';

const Home = ({ onNavigate }) => {
  return (
    <section className="section">
      <div className="container">
        <div className="hero-content">
          <div className="hero-text">
            <h1 className="hero-title">
              Quantum Machine Learning
              <span className="highlight"> FTSE 100 Predictor</span>
            </h1>
            <p className="hero-subtitle">
              Leveraging quantum computing to predict intraday movements of the FTSE 100 index 
              with unprecedented accuracy using variational quantum classifiers.
            </p>
            <div className="hero-stats">
              <div className="stat-item">
                <div className="stat-number">3</div>
                <div className="stat-label">Years of Data</div>
              </div>
              <div className="stat-item">
                <div className="stat-number">5</div>
                <div className="stat-label">Time Intervals</div>
              </div>
              <div className="stat-item">
                <div className="stat-number">8</div>
                <div className="stat-label">Features</div>
              </div>
            </div>
            <button 
              className="cta-button" 
              onClick={() => onNavigate('dashboard')}
            >
              Start Predicting
            </button>
          </div>
          <div className="hero-visual">
            <div className="quantum-animation">
              <div className="quantum-circle"></div>
              <div className="quantum-particles"></div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Home;
