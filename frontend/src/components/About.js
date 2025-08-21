import React from 'react';
import './About.css';

const About = () => {
  return (
    <section className="section about-section">
      <div className="container">
        <h2 className="section-title">Built for signal. Designed for speed.</h2>
        <div className="about-content">
          <div className="about-text">
            <h3>Production‑grade Quantum ML</h3>
            <p>
              We combine classical feature engineering with a Variational Quantum Classifier to
              capture non‑linear structure in FTSE 100 dynamics. The system trains on a sliding
              3‑year window ending yesterday and makes intraday calls at five time points.
            </p>

            <h3>Transparent, reproducible signals</h3>
            <p>
              Signals are derived from 8 interpretable inputs like RSI, MACD, moving averages,
              volatility and volume regimes. All steps are logged and cached for fast reloads.
            </p>

            <h3>Fast path to insights</h3>
            <p>
              The app caches trained results per date and reuses the latest model to deliver
              answers instantly, even after restarts.
            </p>
          </div>
          <div className="about-features">
            <div className="feature-card">
              <div className="feature-icon">⚛</div>
              <h4>Quantum VQC core</h4>
              <p>Parameterized circuits optimized on stratified splits</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">📈</div>
              <h4>8 engineered features</h4>
              <p>Signals distilled from momentum, trend and risk</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h4>Instant reloads</h4>
              <p>CSV cache for historical dates; daily auto‑train for today</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
