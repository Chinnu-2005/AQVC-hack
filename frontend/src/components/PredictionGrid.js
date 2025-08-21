import React from 'react';
import './PredictionGrid.css';

const PredictionGrid = ({ predictions }) => {
  const timeSlots = ['08:00', '10:00', '12:00', '14:00', '16:00'];

  return (
    <div className="predictions-grid">
      <h3>Intraday Predictions</h3>
      <div className="grid-container">
        {timeSlots.map(time => {
          const prediction = predictions[time];
          if (!prediction) return null;

          const movementClass = prediction.movement === 'UP' ? 'movement-up' : 'movement-down';
          const movementIcon = prediction.movement === 'UP' ? '↗' : '↘';

          return (
            <div key={time} className="prediction-card">
              <div className="prediction-time">{time}</div>
              <div className={`prediction-movement ${movementClass}`}>
                {movementIcon} {prediction.movement}
              </div>
              <div className="prediction-probability">
                Probability: {(prediction.probability * 100).toFixed(1)}%
              </div>
              <div className="prediction-confidence">
                Confidence: {(prediction.confidence * 100).toFixed(1)}%
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default PredictionGrid;
