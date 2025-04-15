import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend
} from 'recharts';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function Dashboard() {
  const [predictions, setPredictions] = useState([]);
  const [performance, setPerformance] = useState(null);
  const [loading, setLoading] = useState(false);
  const [symbols, setSymbols] = useState('RELIANCE,TCS,INFY');

  const fetchPredictions = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/predict`, {
        symbols: symbols.split(',').map(s => s.trim())
      });
      setPredictions(response.data);
    } catch (error) {
      console.error('Error fetching predictions:', error);
    }
    setLoading(false);
  };

  const fetchPerformance = async () => {
    try {
      const response = await axios.get(`${API_URL}/performance?days=30`);
      setPerformance(response.data);
    } catch (error) {
      console.error('Error fetching performance:', error);
    }
  };

  useEffect(() => {
    fetchPerformance();
  }, []);

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Stock ML Platform</h1>
      
      {/* Performance Metrics */}
      {performance && (
        <div className="grid grid-cols-4 gap-4 mb-6">
          <MetricCard title="Accuracy" value={`${(performance.accuracy * 100).toFixed(2)}%`} />
          <MetricCard title="Sharpe Ratio" value={performance.sharpe_ratio.toFixed(2)} />
          <MetricCard title="Win Rate" value={`${(performance.win_rate * 100).toFixed(2)}%`} />
          <MetricCard title="Total Predictions" value={performance.total_predictions} />
        </div>
      )}

      {/* Prediction Input */}
      <div className="bg-white shadow rounded-lg p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Get Predictions</h2>
        <div className="flex gap-4">
          <input
            type="text"
            value={symbols}
            onChange={(e) => setSymbols(e.target.value)}
            placeholder="Enter symbols (comma-separated)"
            className="flex-1 border rounded px-4 py-2"
          />
          <button
            onClick={fetchPredictions}
            disabled={loading}
            className="bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600 disabled:bg-gray-400"
          >
            {loading ? 'Loading...' : 'Predict'}
          </button>
        </div>
      </div>

      {/* Predictions Table */}
      {predictions.length > 0 && (
        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Latest Predictions</h2>
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2">Symbol</th>
                <th className="text-left py-2">Signal</th>
                <th className="text-left py-2">Confidence</th>
                <th className="text-left py-2">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((pred, idx) => (
                <tr key={idx} className="border-b">
                  <td className="py-2 font-semibold">{pred.symbol}</td>
                  <td className="py-2">
                    <span className={`px-3 py-1 rounded ${
                      pred.signal === 'BUY' ? 'bg-green-200 text-green-800' :
                      pred.signal === 'SELL' ? 'bg-red-200 text-red-800' :
                      'bg-gray-200 text-gray-800'
                    }`}>
                      {pred.signal}
                    </span>
                  </td>
                  <td className="py-2">{(pred.confidence * 100).toFixed(1)}%</td>
                  <td className="py-2">{new Date(pred.timestamp).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function MetricCard({ title, value }) {
  return (
    <div className="bg-white shadow rounded-lg p-4">
      <div className="text-sm text-gray-600 mb-1">{title}</div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  );
}

export default Dashboard;