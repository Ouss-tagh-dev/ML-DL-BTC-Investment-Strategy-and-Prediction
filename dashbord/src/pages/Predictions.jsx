import React, { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { modelsApi, dataApi } from "../services/api";
import { Play, AlertCircle, Cpu, TrendingUp, TrendingDown, Clock, Search } from "lucide-react";

export default function Predictions() {
  const [results, setResults] = useState(null);

  const batch = useMutation({
    mutationFn: (p) => modelsApi.batchPredict(p).then((r) => r.data),
  });

  const featuresQ = useQuery({
    queryKey: ["feature-names"],
    queryFn: () => dataApi.getFeatureNames().then((r) => r.features),
  });

  const handleBatch = React.useCallback(async () => {
    try {
      const res = await batch.mutateAsync({ use_latest: true });
      setResults(res);
    } catch (e) {
      setResults({ error: e.message });
    }
  }, [batch]);

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-8">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 bg-gray-800/20 p-6 rounded-3xl border border-gray-700/30">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Strategy Inference</h1>
          <p className="text-gray-400 mt-1">Run all models against the most recent market snapshot.</p>
        </div>
        <button
          onClick={handleBatch}
          disabled={batch.isPending}
          className="flex items-center justify-center space-x-2 px-8 py-4 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 text-white rounded-2xl transition-all shadow-xl shadow-blue-900/20 active:scale-95 font-bold"
        >
          {batch.isPending ? (
            <div className="w-5 h-5 border-3 border-white border-t-transparent rounded-full animate-spin" />
          ) : (
            <Play size={20} fill="currentColor" />
          )}
          <span>{batch.isPending ? "Computing..." : "Run Multi-Model Prediction"}</span>
        </button>
      </div>

      {results?.predictions ? (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-lg text-gray-200">Ensemble Results</h3>
            <span className="text-xs text-gray-500 bg-gray-800 px-3 py-1 rounded-full border border-gray-700">
              {results.count} models processed
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {results.predictions.map((p) => (
              <PredictionResultCard key={p.model_id} result={p} />
            ))}
          </div>
        </div>
      ) : results?.error ? (
        <div className="p-10 bg-red-500/10 border border-red-500/30 rounded-3xl flex flex-col items-center justify-center text-center">
          <AlertCircle size={48} className="text-red-500 mb-4" />
          <h3 className="text-xl font-bold text-white">Inference Failed</h3>
          <p className="text-red-300/80 max-w-md mt-2">{results.error}</p>
        </div>
      ) : (
        <div className="p-20 bg-gray-800/20 border border-dashed border-gray-700 rounded-3xl flex flex-col items-center justify-center text-center">
          <div className="p-4 bg-gray-800 rounded-2xl mb-4">
            <Clock size={32} className="text-gray-500" />
          </div>
          <h3 className="text-xl font-bold text-white">No active inference</h3>
          <p className="text-gray-500 max-w-sm mt-2">Click the button above to start a batch prediction using the latest feature set.</p>
        </div>
      )}

      {/* Feature Preview Section */}
      <div className="bg-gray-800/40 border border-gray-700/50 p-6 rounded-3xl shadow-lg">
        <div className="flex items-center space-x-2 text-gray-300 mb-6">
          <Search size={18} className="text-blue-500" />
          <h3 className="font-semibold uppercase tracking-wider text-sm">Feature Engineering Preview</h3>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
          {featuresQ.data?.slice(0, 18).map(feat => (
            <div key={feat} className="text-[10px] bg-gray-900/60 p-2 rounded-lg border border-gray-700/30 text-gray-400 font-mono text-center truncate">
              {feat}
            </div>
          ))}
          {featuresQ.data?.length > 18 && (
            <div className="text-[10px] bg-gray-800 p-2 rounded-lg border border-gray-700/30 text-gray-500 text-center font-bold italic">
              +{featuresQ.data.length - 18} more
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function PredictionResultCard({ result }) {
  const isUp = result.direction === 'UP';
  return (
    <div className={`p-5 rounded-2xl border transition-all ${isUp ? 'bg-green-500/5 border-green-500/20 hover:border-green-500/50' : 'bg-red-500/5 border-red-500/20 hover:border-red-500/50'}`}>
      <div className="flex justify-between items-start mb-4">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-xl ${isUp ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>
            <Cpu size={18} />
          </div>
          <div>
            <div className="font-bold text-white text-sm leading-tight">{result.model_id.replace(/_/g, ' ').toUpperCase()}</div>
            <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest mt-0.5">Prediction</div>
          </div>
        </div>
        {isUp ? <TrendingUp className="text-green-500" size={20} /> : <TrendingDown className="text-red-500" size={20} />}
      </div>

      <div className="flex items-end justify-between">
        <div className={`text-2xl font-black ${isUp ? 'text-green-500' : 'text-red-500'}`}>
          {result.direction}
        </div>
        <div className="text-right">
          <div className="text-xs text-gray-500 font-medium -mb-1">Confidence</div>
          <div className="text-lg font-bold text-white">{(result.confidence * 100).toFixed(1)}%</div>
        </div>
      </div>

      {/* Mini Confidence bar */}
      <div className="mt-4 h-1.5 w-full bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full transition-all duration-1000 ${isUp ? 'bg-green-500' : 'bg-red-500'}`}
          style={{ width: `${result.confidence * 100}%` }}
        />
      </div>
    </div>
  );
}
