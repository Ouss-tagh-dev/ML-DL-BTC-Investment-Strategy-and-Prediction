import React, { useState, useEffect, useCallback } from "react";
import PriceChart from "../components/charts/PriceChart";
import { useHistoricalData, useModels, useLatestData, useBatchPredict } from "../hooks/useApi";
import { useQuery } from "@tanstack/react-query";
import { metricsApi } from "../services/api";
import { TrendingUp, TrendingDown, Activity, BarChart3, Clock, Zap, AlertCircle, BrainCircuit } from "lucide-react";

export default function Dashboard() {
  const [consensus, setConsensus] = useState({ signal: "WAIT", ratio: 0, count: 0, upCount: 0, details: [] });
  const histQ = useHistoricalData({ limit: 168 });
  const modelsQ = useModels();
  const latestQ = useLatestData(1, { refetchInterval: 5000 });
  const batchPredict = useBatchPredict();
  const { mutateAsync } = batchPredict;

  const perfQ = useQuery({
    queryKey: ["performance"],
    queryFn: () => metricsApi.getPerformance().then((r) => r.data),
  });

  const handleInference = useCallback(async () => {
    try {
      const res = await mutateAsync({ use_latest: true });
      if (res.predictions) {
        const upCount = res.predictions.filter(p => p.direction === "UP").length;
        const total = res.predictions.length;
        const ratio = total > 0 ? upCount / total : 0;

        setConsensus({
          signal: ratio > 0.6 ? "BUY" : ratio < 0.4 ? "SELL" : "NEUTRAL",
          ratio: ratio,
          count: total,
          upCount: upCount,
          details: res.predictions
        });
      }
    } catch (e) {
      console.error("Inference failed", e);
    }
  }, [mutateAsync]);

  // Run initial inference
  useEffect(() => {
    handleInference();
  }, [handleInference]);

  const histArray = histQ.data?.data || [];
  const chartData = histArray.map((d) => ({
    time: d.timestamp || d.time || d.t || d.date || new Date().toISOString(),
    open: d.Open || d.open || 0,
    high: d.High || d.high || 0,
    low: d.Low || d.low || 0,
    close: d.Close || d.close || 0,
  }));

  const latestPrice = latestQ.data?.price ?? 0;
  const change24h = latestQ.data?.change24h ?? 0;
  const metrics = perfQ.data?.metrics || [];

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6 alpha-overlay">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Market Hub</h1>
          <p className="text-gray-400">Integrated intelligence for Bitcoin directional strategies</p>
        </div>
        <div className="flex items-center space-x-2 text-xs text-gray-500 bg-gray-800/50 px-3 py-1.5 rounded-full border border-gray-700">
          <Clock size={14} />
          <span>Last live ticker: {new Date().toLocaleTimeString()}</span>
        </div>
      </div>

      {/* Top Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Live BTC Price"
          value={`$${latestPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}`}
          subValue={<PriceChange change={change24h} />}
          icon={<Activity className="text-blue-400" />}
        />
        <StatCard
          label="Volatility (24h)"
          value={`${change24h.toFixed(2)}%`}
          subValue={change24h >= 0 ? "Momentum: Positif" : "Momentum: Négatif"}
          icon={change24h >= 0 ? <TrendingUp className="text-green-400" /> : <TrendingDown className="text-red-400" />}
        />
        <StatCard
          label="Neural Models"
          value={(modelsQ.data || []).length}
          subValue={`${(modelsQ.data || []).filter(m => m.loaded).length} engine(s) optimized`}
          icon={<BarChart3 className="text-purple-400" />}
        />
        <StatCard
          label="Top Accuracy"
          value={`${((Math.max(...metrics.map(m => m.accuracy || 0)) || 0.52) * 100).toFixed(1)}%`}
          subValue="Ensemble record"
          icon={<TrendingUp className="text-yellow-400" />}
        />
      </div>

      {/* Main Chart */}
      <div className="bg-gray-800/40 border border-gray-700/50 backdrop-blur-sm p-5 rounded-2xl shadow-xl">
        <div className="flex items-center justify-between mb-6">
          <h3 className="font-semibold text-lg flex items-center space-x-2">
            <Activity size={20} className="text-blue-500" />
            <span>Market Trajectory (7D)</span>
          </h3>
          <div className="flex space-x-2">
            <span className="bg-blue-500/10 text-blue-400 text-xs px-2.5 py-1 rounded-md border border-blue-500/20 font-bold uppercase tracking-widest">Binance Data</span>
          </div>
        </div>
        <PriceChart data={chartData} height={400} />
      </div>

      {/* Grid for consensus and performance */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 bg-gray-800/40 border border-gray-700/50 p-6 rounded-2xl space-y-4 flex flex-col">
          <h3 className="font-semibold text-lg flex items-center space-x-2">
            <BrainCircuit size={18} className="text-indigo-400" />
            <span>AI Consensus</span>
          </h3>

          <div className="flex-1 flex flex-col items-center justify-center space-y-4 bg-gray-900/40 p-10 rounded-2xl border border-dashed border-gray-700 relative overflow-hidden group">
            {batchPredict.isPending && (
              <div className="absolute inset-0 bg-gray-950/60 backdrop-blur-sm flex items-center justify-center z-10 transition-opacity">
                <div className="flex flex-col items-center space-y-3">
                  <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
                  <span className="text-[10px] font-bold text-blue-400 uppercase tracking-widest">Processing Ensemble...</span>
                </div>
              </div>
            )}

            <div className={`text-5xl font-black tracking-tighter ${consensus.signal === "BUY" ? "text-green-500" :
              consensus.signal === "SELL" ? "text-red-500" : "text-gray-400"
              }`}>
              {consensus.signal}
            </div>
            <p className="text-gray-400 text-xs text-center font-medium leading-relaxed">
              {consensus.count > 0
                ? `Analysis complete across ${consensus.count} models. Consensus: ${consensus.upCount}/${consensus.count} expect an upward move.`
                : "Initialize system to reach a consensus across the platform models."}
            </p>

            {consensus.count > 0 && (
              <div className="w-full space-y-4">
                <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden border border-gray-700/50">
                  <div
                    className={`h-full transition-all duration-1000 ${consensus.ratio > 0.5 ? 'bg-green-500' : 'bg-red-500'}`}
                    style={{ width: `${consensus.ratio * 100}%` }}
                  />
                </div>

                {/* Model Breakdown */}
                <div className="bg-gray-950/30 rounded-xl p-3 border border-gray-800/50 max-h-40 overflow-y-auto custom-scrollbar">
                  <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-2 sticky top-0 bg-gray-950/90 backdrop-blur-sm p-1">
                    Voting Breakdown
                  </h4>
                  <div className="grid grid-cols-2 gap-2">
                    {consensus.details?.map((p, i) => (
                      <div key={i} className="flex items-center justify-between bg-gray-900/50 px-2 py-1.5 rounded-lg border border-gray-800">
                        <span className="text-xs font-medium text-gray-300 truncate max-w-[80px]" title={p.model_id}>
                          {p.model_id?.replace(/_/g, ' ') || `Model ${i + 1}`}
                        </span>
                        <span className={`text-[10px] font-bold px-1.5 rounded ${p.direction === "UP" ? "text-green-400 bg-green-500/10" : "text-red-400 bg-red-500/10"
                          }`}>
                          {p.direction}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          <button
            onClick={handleInference}
            disabled={batchPredict.isPending}
            className="w-full py-3.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 text-white rounded-xl transition-all font-bold shadow-lg shadow-blue-900/20 flex items-center justify-center space-x-2 active:scale-[0.98]"
          >
            <Zap size={18} />
            <span>RUN REAL-TIME INFERENCE</span>
          </button>
        </div>

        <div className="lg:col-span-2 bg-gray-800/40 border border-gray-700/50 p-6 rounded-2xl">
          <div className="flex items-center justify-between mb-6">
            <h3 className="font-semibold text-lg flex items-center space-x-2">
              <BarChart3 size={18} className="text-blue-400" />
              <span>Performance Alpha</span>
            </h3>
            <button className="text-[10px] font-bold text-gray-400 hover:text-white uppercase tracking-widest transition-colors flex items-center space-x-1">
              <span>View All</span>
              <Activity size={12} />
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {metrics.length > 0 ? metrics.slice(0, 4).map((m) => (
              <div key={m.model_id} className="bg-gray-900/40 p-4 rounded-xl border border-gray-700/30 flex justify-between items-center group hover:border-blue-500/50 transition-all hover:bg-gray-900/60">
                <div>
                  <div className="text-sm font-bold text-white group-hover:text-blue-400 transition-colors uppercase tracking-tight">{m.model_name}</div>
                  <div className="text-[10px] text-gray-500 font-bold uppercase tracking-[0.1em]">{m.model_id}</div>
                </div>
                <div className="text-right">
                  <div className="text-lg font-black text-blue-400 leading-none">{(m.accuracy * 100).toFixed(1)}%</div>
                  <div className="text-[9px] text-gray-500 uppercase tracking-widest font-bold">Accuracy</div>
                </div>
              </div>
            )) : (
              <div className="col-span-2 py-10 flex flex-col items-center justify-center text-gray-600 italic text-sm">
                <AlertCircle size={24} className="mb-2 opacity-20" />
                Initializing metrics database...
              </div>
            )}
          </div>
        </div>
      </div >
    </div >
  );
}

function StatCard({ label, value, subValue, icon }) {
  return (
    <div className="bg-gray-800/40 border border-gray-700/50 backdrop-blur-sm p-5 rounded-2xl shadow-lg group hover:bg-gray-800/60 transition-all">
      <div className="flex justify-between items-start mb-2">
        <span className="text-gray-400 text-sm font-medium">{label}</span>
        <div className="p-2 bg-gray-900/50 rounded-xl group-hover:scale-110 transition-transform">
          {icon}
        </div>
      </div>
      <div className="text-2xl font-bold text-white mb-1">{value}</div>
      <div className="text-sm text-gray-500">{subValue}</div>
    </div>
  );
}

function PriceChange({ change }) {
  const isPositive = change >= 0;
  return (
    <div className={`flex items-center space-x-1 ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
      {isPositive ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
      <span className="font-medium">{isPositive ? '+' : ''}{change.toFixed(2)}%</span>
    </div>
  );
}
