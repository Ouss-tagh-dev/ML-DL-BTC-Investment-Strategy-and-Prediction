import React, { useEffect, useState } from "react";
import { modelsApi } from "../services/api";
import { X, ShieldCheck, Database, Layout, Target, TrendingUp, Info } from "lucide-react";

export default function ModelDetailsModal({ modelId, onClose }) {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!modelId) return;
    setLoading(true);
    let mounted = true;
    modelsApi
      .getInfo(modelId)
      .then((r) => {
        if (mounted) {
          setInfo(r.data);
          setLoading(false);
        }
      })
      .catch(() => {
        if (mounted) setLoading(false);
      });
    return () => { mounted = false; };
  }, [modelId]);

  if (!modelId) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-700/50 text-gray-100 max-w-3xl w-full max-h-[90vh] overflow-hidden rounded-3xl shadow-2xl flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-gray-800 flex justify-between items-center bg-gray-800/20">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-500/10 text-blue-400 rounded-xl">
              <Info size={24} />
            </div>
            <div>
              <h3 className="text-xl font-bold">{info?.name || "Model Details"}</h3>
              <p className="text-xs text-gray-500 font-medium uppercase tracking-widest">{modelId}</p>
            </div>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-gray-800 rounded-full transition-colors">
            <X size={24} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto custom-scrollbar flex-1">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-20 space-y-4">
              <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
              <p className="text-gray-400 font-medium">Fetching metadata...</p>
            </div>
          ) : info ? (
            <div className="space-y-8">
              {/* Stats Overview */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MiniStat label="Accuracy" value={`${(info.accuracy * 100).toFixed(1)}%`} icon={<Target size={14} />} />
                <MiniStat label="Sharpe" value={info.sharpe_ratio?.toFixed(2) || "N/A"} icon={<TrendingUp size={14} />} />
                <MiniStat label="Status" value={info.status || "Standing by"} icon={<ShieldCheck size={14} />} />
                <MiniStat label="Format" value={info.architecture?.layers ? "DL/Keras" : "ML/Sklearn"} icon={<Database size={14} />} />
              </div>

              {/* Architecture Section */}
              <Section title="Architecture" icon={<Layout size={18} />}>
                {info.architecture?.layers ? (
                  <ul className="space-y-2">
                    {info.architecture.layers.map((layer, idx) => (
                      <li key={idx} className="bg-gray-800/40 p-3 rounded-xl border border-gray-700/30 text-sm font-mono text-blue-300">
                        {layer}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="bg-gray-800/40 p-4 rounded-xl border border-gray-700/30 text-gray-300 text-sm">
                    {info.architecture || "Standard statistical machine learning architecture."}
                  </div>
                )}
              </Section>

              {/* Config Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Section title="Training Config" icon={<ShieldCheck size={18} />}>
                  <div className="bg-gray-800/40 p-4 rounded-xl border border-gray-700/30">
                    <pre className="text-xs text-blue-200/70 font-mono overflow-x-auto">
                      {JSON.stringify(info.training_config || info.training || {}, null, 2)}
                    </pre>
                  </div>
                </Section>
                <Section title="Data Source" icon={<Database size={18} />}>
                  <div className="bg-gray-800/40 p-4 rounded-xl border border-gray-700/30">
                    <div className="text-sm text-gray-400 mb-2">Features used: <span className="text-blue-400">{(info.data_config?.number_of_features || info.data_info?.num_features || 30)}</span></div>
                    <div className="flex flex-wrap gap-1.5">
                      {['BTC/USDT', 'OHLCV', 'RSI', 'MACD'].map(tag => (
                        <span key={tag} className="text-[10px] bg-gray-700 text-gray-300 px-2 py-0.5 rounded uppercase font-bold">{tag}</span>
                      ))}
                    </div>
                  </div>
                </Section>
              </div>
            </div>
          ) : (
            <div className="text-center py-20 text-red-400">Failed to load model metadata.</div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-800 bg-gray-800/10 flex justify-end">
          <button
            onClick={onClose}
            className="px-6 py-2 bg-gray-700 hover:bg-gray-600 text-white font-semibold rounded-xl transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

function Section({ title, icon, children }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center space-x-2 text-gray-400">
        {icon}
        <h4 className="font-semibold text-sm uppercase tracking-wider">{title}</h4>
      </div>
      {children}
    </div>
  );
}

function MiniStat({ label, value, icon }) {
  return (
    <div className="bg-gray-800/40 p-3 rounded-2xl border border-gray-700/30">
      <div className="flex items-center space-x-1.5 text-gray-500 mb-1">
        {icon}
        <span className="text-[10px] font-bold uppercase tracking-tighter">{label}</span>
      </div>
      <div className="text-sm font-bold text-white">{value}</div>
    </div>
  );
}
