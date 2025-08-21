import React from 'react';
import './ActualDataTable.css';

const ActualDataTable = ({ actualData }) => {
  return (
    <div className="actual-data-section">
      <h3>Actual Market Data</h3>
      <div className="data-table">
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Open</th>
              <th>High</th>
              <th>Low</th>
              <th>Close</th>
              <th>Volume</th>
              <th>Returns</th>
            </tr>
          </thead>
          <tbody>
            {actualData.target_data && (
              <tr className="target-row">
                <td><strong>{actualData.target_data.date}</strong></td>
                <td>{actualData.target_data.open.toFixed(2)}</td>
                <td>{actualData.target_data.high.toFixed(2)}</td>
                <td>{actualData.target_data.low.toFixed(2)}</td>
                <td>{actualData.target_data.close.toFixed(2)}</td>
                <td>{actualData.target_data.volume.toLocaleString()}</td>
                <td>{(actualData.target_data.returns * 100).toFixed(2)}%</td>
              </tr>
            )}
            {actualData.context_data.map((dataPoint, index) => (
              <tr key={index}>
                <td>{dataPoint.date}</td>
                <td>{dataPoint.open.toFixed(2)}</td>
                <td>{dataPoint.high.toFixed(2)}</td>
                <td>{dataPoint.low.toFixed(2)}</td>
                <td>{dataPoint.close.toFixed(2)}</td>
                <td>{dataPoint.volume.toLocaleString()}</td>
                <td>{(dataPoint.returns * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ActualDataTable;
