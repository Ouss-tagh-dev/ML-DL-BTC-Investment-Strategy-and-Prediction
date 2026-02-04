import React from "react";
import { useQuery } from "@tanstack/react-query";
import { comparisonApi } from "../services/api";
import MetricsChart from "../components/charts/MetricsChart";
import { Info, TrendingUp, Award } from "lucide-react";

export default function Comparison() {
  const accQ = useQuery({
    queryKey: ["comparison", "accuracy"],
    queryFn: () => comparisonApi.getAccuracy().then((r) => r.data),
  });
  const sharpeQ = useQuery({
    queryKey: ["comparison", "sharpe"],
    queryFn: () => comparisonApi.getSharpeRatio().then((r) => r.data),
  });

  const accuracyData = accQ.data?.data || [];
  const sharpeData = sharpeQ.data?.data || [];

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Model Comparison</h1>
          <p className="text-gray-400">Head-to-head analysis of ML and DL strategies</p>
        </div>
        <div className="flex items-center space-x-2 bg-blue-500/10 text-blue-400 px-4 py-2 rounded-xl border border-blue-500/20">
          <Award size={18} />
          <span className="font-semibold">{accuracyData[0]?.model || "Loading..."} leads in Accuracy</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800/40 border border-gray-700/50 p-6 rounded-2xl shadow-xl">
          <div className="flex items-center justify-between mb-6">
            <h3 className="font-semibold text-lg flex items-center space-x-2">
              <TrendingUp size={20} className="text-green-500" />
              <span>Accuracy Distribution</span>
            </h3>
            <Tooltip text="Measures the percentage of correct directional predictions." />
          </div>
          <MetricsChart
            data={accuracyData}
            metric="accuracy"
            height={350}
          />
        </div>

        <div className="bg-gray-800/40 border border-gray-700/50 p-6 rounded-2xl shadow-xl">
          <div className="flex items-center justify-between mb-6">
            <h3 className="font-semibold text-lg flex items-center space-x-2">
              <Award size={20} className="text-yellow-500" />
              <span>Sharpe Ratio Comparison</span>
            </h3>
            <Tooltip text="Risk-adjusted return metric. Higher is better." />
          </div>
          <MetricsChart
            data={sharpeData}
            metric="sharpe_ratio"
            height={350}
          />
        </div>
      </div>

      <div className="bg-blue-900/20 border border-blue-500/30 p-4 rounded-xl flex items-start space-x-3">
        <Info className="text-blue-400 mt-0.5 shrink-0" size={20} />
        <p className="text-sm text-blue-200/80 leading-relaxed">
          Deep Learning models (LSTM, GRU) generally show higher sequence sensitivity, while traditional ML models like Random Forest maintain robust performance on historical statistical features.
        </p>
      </div>
    </div>
  );
}

function Tooltip({ text }) {
  return (
    <div className="relative group">
      <Info size={16} className="text-gray-500 cursor-help hover:text-gray-300 transition-colors" />
      <div className="absolute bottom-full right-0 mb-2 w-48 p-2 bg-gray-900 text-[10px] text-gray-300 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none border border-gray-700 shadow-2xl z-50">
        {text}
      </div>
    </div>
  );
}
