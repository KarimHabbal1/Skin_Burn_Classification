import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function PerformanceComparison() {
  // Sample data - in a real application, this would come from your backend
  const epochs = Array.from({length: 100}, (_, i) => i + 1);
  
  const data = {
    labels: epochs,
    datasets: [
      {
        label: 'CNN Loss',
        data: epochs.map(x => 2.5 * Math.exp(-0.05 * x) + 0.1),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      },
      {
        label: 'QCNN Loss',
        data: epochs.map(x => 2.0 * Math.exp(-0.07 * x) + 0.05),
        borderColor: 'rgb(255, 99, 132)',
        tension: 0.1
      }
    ]
  };

  const accuracyData = {
    labels: epochs,
    datasets: [
      {
        label: 'CNN Accuracy',
        data: epochs.map(x => 0.85 + 0.1 * (1 - Math.exp(-0.05 * x))),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      },
      {
        label: 'QCNN Accuracy',
        data: epochs.map(x => 0.88 + 0.08 * (1 - Math.exp(-0.07 * x))),
        borderColor: 'rgb(255, 99, 132)',
        tension: 0.1
      }
    ]
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Model Performance Comparison'
      }
    },
    scales: {
      y: {
        beginAtZero: true
      }
    }
  };

  return (
    <div className="performance-comparison">
      <h3>Model Performance Analysis</h3>
      
      <div className="chart-container">
        <div className="chart">
          <h4>Training Loss</h4>
          <Line data={data} options={options} />
        </div>
        
        <div className="chart">
          <h4>Training Accuracy</h4>
          <Line data={accuracyData} options={options} />
        </div>
      </div>

      <div className="performance-metrics">
        <div className="metric-card">
          <h4>CNN Model</h4>
          <ul>
            <li>Final Accuracy: 92.5%</li>
            <li>Training Time: 45 minutes</li>
            <li>Parameters: 1.2M</li>
          </ul>
        </div>
        
        <div className="metric-card">
          <h4>Quantum CNN Model</h4>
          <ul>
            <li>Final Accuracy: 94.8%</li>
            <li>Training Time: 60 minutes</li>
            <li>Parameters: 1.5M</li>
            <li>Quantum Qubits: 4</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default PerformanceComparison; 