import React from 'react';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import './PredictionCharts.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const PredictionCharts = ({ predictions }) => {
  const timeSlots = ['08:00', '10:00', '12:00', '14:00', '16:00'];
  
  const movements = timeSlots.map(time => {
    const pred = predictions[time];
    return pred ? (pred.movement === 'UP' ? 1 : 0) : 0;
  });

  const confidences = timeSlots.map(time => {
    const pred = predictions[time];
    return pred ? pred.confidence * 100 : 0;
  });

  const predictionChartData = {
    labels: timeSlots,
    datasets: [
      {
        label: 'Predicted Movement',
        data: movements,
        borderColor: '#00d4ff',
        backgroundColor: 'rgba(0, 212, 255, 0.1)',
        borderWidth: 3,
        fill: true,
        tension: 0.4,
      },
    ],
  };

  const confidenceChartData = {
    labels: timeSlots,
    datasets: [
      {
        label: 'Confidence Level (%)',
        data: confidences,
        backgroundColor: 'rgba(0, 212, 255, 0.6)',
        borderColor: '#00d4ff',
        borderWidth: 2,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: '#ffffff',
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          color: '#ffffff',
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
      },
      x: {
        ticks: {
          color: '#ffffff',
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
      },
    },
  };

  const predictionChartOptions = {
    ...chartOptions,
    scales: {
      ...chartOptions.scales,
      y: {
        ...chartOptions.scales.y,
        max: 1,
        ticks: {
          ...chartOptions.scales.y.ticks,
          stepSize: 1,
          callback: function(value) {
            return value === 1 ? 'UP' : 'DOWN';
          },
        },
      },
    },
  };

  const confidenceChartOptions = {
    ...chartOptions,
    scales: {
      ...chartOptions.scales,
      y: {
        ...chartOptions.scales.y,
        max: 100,
      },
    },
  };

  return (
    <div className="charts-container">
      <div className="chart-section">
        <h3>Predicted vs Actual Movement</h3>
        <div className="chart-wrapper">
          <Line data={predictionChartData} options={predictionChartOptions} />
        </div>
      </div>
      <div className="chart-section">
        <h3>Confidence Levels</h3>
        <div className="chart-wrapper">
          <Bar data={confidenceChartData} options={confidenceChartOptions} />
        </div>
      </div>
    </div>
  );
};

export default PredictionCharts;
