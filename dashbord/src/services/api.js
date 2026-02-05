import axios from "axios";

const API_BASE_URL = "http://localhost:8000/api";

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export const systemApi = {
  getHealth: () => axios.get("http://localhost:8000/health"),
};

export const dataApi = {
  getHistorical: (params) => api.get("/data/historical", { params }),
  getLatest: (n = 1) => api.get(`/data/latest?n=${n}`),
  getStatistics: () => api.get("/data/statistics"),
  getFeatureNames: () => api.get("/data/feature-names"),
};

export const modelsApi = {
  list: () => api.get("/models/list"),
  predict: (data) => api.post("/models/predict", data),
  batchPredict: (data) => api.post("/models/batch-predict", data),
  getInfo: (modelId) => api.get(`/models/${modelId}/info`),
};

export const metricsApi = {
  getPerformance: () => api.get("/metrics/performance"),
  getBacktesting: (modelId) => api.get(`/metrics/${modelId}/backtesting`),
  getComparison: () => api.get("/metrics/comparison"),
};

export const comparisonApi = {
  getAccuracy: () => api.get("/comparison/accuracy"),
  getSharpeRatio: () => api.get("/comparison/sharpe-ratio"),
  getReturns: () => api.get("/comparison/returns"),
  getAllMetrics: () => api.get("/comparison/all-metrics"),
  getMlVsDl: () => api.get("/comparison/ml-vs-dl"),
};

export default api;
