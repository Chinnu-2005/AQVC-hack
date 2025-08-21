import React, { useState, useEffect } from 'react';
import './Dashboard.css';
import PredictionGrid from './PredictionGrid';
import PredictionCharts from './PredictionCharts';
import ActualDataTable from './ActualDataTable';
import LoadingSpinner from './LoadingSpinner';

const Dashboard = () => {
  const [selectedDate, setSelectedDate] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [predictionData, setPredictionData] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    // Set today's date as default
    const today = new Date().toISOString().split('T')[0];
    setSelectedDate(today);
  }, []);

  const handleDateChange = (e) => {
    setSelectedDate(e.target.value);
  };

  const getPredictions = async () => {
    if (!selectedDate) {
      setError('Please select a date');
      return;
    }

    setIsLoading(true);
    setError('');
    setPredictionData(null);

    try {
      const today = new Date().toISOString().split('T')[0];
      const isToday = selectedDate === today;
      
      let response;
      if (isToday) {
        response = await fetch('/model/predict/latest', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          }
        });
      } else {
        response = await fetch(`/model/date?target_date=${selectedDate}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          }
        });
      }

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setPredictionData(data);
      
    } catch (error) {
      console.error('Error fetching predictions:', error);
      setError('Error fetching predictions. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <section className="section">
      <div className="container">
        <h2 className="section-title">Prediction Dashboard</h2>
        
        {/* Date Selection */}
        <div className="date-selector">
          <label htmlFor="dateInput">Select Date:</label>
          <input 
            type="date" 
            id="dateInput" 
            className="date-input"
            value={selectedDate}
            onChange={handleDateChange}
          />
          <button 
            className="predict-button"
            onClick={getPredictions}
            disabled={isLoading}
          >
            {isLoading ? 'Getting Predictions...' : 'Get Predictions'}
          </button>
        </div>

        {/* Error Message */}
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {/* Loading State */}
        {isLoading && <LoadingSpinner />}

        {/* Results */}
        {predictionData && !isLoading && (
          <div className="results-container">
            {/* Prediction Summary */}
            <div className="prediction-summary">
              <h3>Prediction Summary</h3>
              <div className="summary-stats">
                <div className="summary-item">
                  <span className="summary-label">Target Date:</span>
                  <span className="summary-value">
                    {predictionData.target_date || new Date().toLocaleDateString()}
                  </span>
                </div>
                <div className="summary-item">
                  <span className="summary-label">Training Period:</span>
                  <span className="summary-value">
                    {predictionData.training_period || '3 years ending yesterday'}
                  </span>
                </div>
                <div className="summary-item">
                  <span className="summary-label">Model Accuracy:</span>
                  <span className="summary-value">
                    {predictionData.training_accuracy 
                      ? `${(predictionData.training_accuracy * 100).toFixed(2)}%`
                      : 'N/A (Auto-trained)'
                    }
                  </span>
                </div>
              </div>
            </div>

            {/* Predictions Grid */}
            <PredictionGrid predictions={predictionData.predictions} />

            {/* Charts */}
            <PredictionCharts predictions={predictionData.predictions} />

            {/* Actual Data Table */}
            {predictionData.actual_data && predictionData.actual_data.data_available && (
              <ActualDataTable actualData={predictionData.actual_data} />
            )}
          </div>
        )}
      </div>
    </section>
  );
};

export default Dashboard;
