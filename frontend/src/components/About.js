import React from 'react';
import './About.css';

const About = () => {
  return (
    <section className="section">
      <div className="container">
        <h2 className="section-title">About Our Quantum Approach</h2>
        <div className="about-content">
          <div className="about-text">
            <h3>Revolutionary Technology</h3>
            <p>
              Our system combines the power of quantum computing with machine learning to analyze 
              FTSE 100 market patterns. Using Variational Quantum Classifiers (VQC), we process 
              complex market data through quantum circuits to generate accurate intraday predictions.
            </p>
            
            <h3>Technical Indicators</h3>
            <p>
              We analyze 8 key technical indicators including RSI, MACD, moving averages, 
              volatility, and volume patterns to create comprehensive market insights.
            </p>
            
            <h3>Real-Time Predictions</h3>
            <p>
              Get predictions for 2-hour intervals throughout the trading day (8:00 AM to 4:00 PM), 
              helping you make informed trading decisions with quantum-powered accuracy.
            </p>
          </div>
          <div className="about-features">
            <div className="feature-card">
              <div className="feature-icon">⚛</div>
              <h4>Quantum Processing</h4>
              <p>Advanced quantum algorithms for pattern recognition</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">📊</div>
              <h4>Technical Analysis</h4>
              <p>Comprehensive market indicators and metrics</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">⏰</div>
              <h4>Intraday Focus</h4>
              <p>2-hour interval predictions for precise timing</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
