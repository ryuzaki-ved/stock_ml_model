import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = {
  predict: (symbols) => axios.post(`${API_URL}/predict`, { symbols }),
  performance: (days = 30) => axios.get(`${API_URL}/performance`, { params: { days } }),
  history: (params = {}) => axios.get(`${API_URL}/predictions/history`, { params }),
};

export default api;

