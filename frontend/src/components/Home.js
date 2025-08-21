import React from 'react';
import './Home.css';

const Home = ({ onNavigate }) => {
  return (
    <section className="section home-section">
      <div className="hero-bg">
        <div className="bg-glow bg-glow-1" />
        <div className="bg-glow bg-glow-2" />
      </div>
      <div className="container">
        <div className="hero-content">
          <div className="hero-text">
            <p className="eyebrow">FTSE 100 • Quantum ML</p>
            <h1 className="hero-title">
              Predict the market with
              <span className="highlight"> Quantum precision</span>
            </h1>
            <p className="hero-subtitle">
              A production-ready quantum ML playground that forecasts intraday moves at
              2‑hour intervals. Built on Variational Quantum Classifiers and engineered for speed.
            </p>
            <div className="cta-row">
              <button
                className="cta-button primary-cta"
                onClick={() => onNavigate('dashboard')}
              >
                Get Predictions
              </button>
              <button
                className="cta-button secondary-cta"
                onClick={() => onNavigate('about')}
              >
                Learn More
              </button>
            </div>
            <div className="hero-stats">
              <div className="stat-item">
                <div className="stat-number">3y</div>
                <div className="stat-label">Sliding window</div>
              </div>
              <div className="stat-item">
                <div className="stat-number">5×</div>
                <div className="stat-label">Intraday slots</div>
              </div>
              <div className="stat-item">
                <div className="stat-number">8</div>
                <div className="stat-label">Engineered features</div>
              </div>
            </div>
          </div>
          <div className="hero-visual">
            <div className="quantum-animation">
              <div className="quantum-circle" />
              <div className="quantum-particles" />
              <div className="orb orb-1" />
              <div className="orb orb-2" />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Home;
