import React, { useState, useEffect, useRef, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { logsApi } from "../services/api";
import {
    Terminal,
    Search,
    Trash2,
    RefreshCw,
    Filter,
    AlertTriangle,
    Info,
    Bug,
    AlertCircle,
    ChevronDown,
    Pause,
    Play,
} from "lucide-react";

const LEVEL_CONFIG = {
    DEBUG: { color: "text-sky-400", bg: "bg-sky-500/10", border: "border-sky-500/30", icon: Bug },
    INFO: { color: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/30", icon: Info },
    WARNING: { color: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/30", icon: AlertTriangle },
    ERROR: { color: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/30", icon: AlertCircle },
};

export default function Logs() {
    const [search, setSearch] = useState("");
    const [levelFilter, setLevelFilter] = useState(null);
    const [autoRefresh, setAutoRefresh] = useState(true);
    const [autoScroll, setAutoScroll] = useState(true);
    const bottomRef = useRef(null);
    const queryClient = useQueryClient();

    const logsQ = useQuery({
        queryKey: ["logs", levelFilter, search],
        queryFn: () =>
            logsApi
                .getLogs({ level: levelFilter, search: search || undefined, limit: 500 })
                .then((r) => r.data),
        refetchInterval: autoRefresh ? 3000 : false,
    });

    const clearMut = useMutation({
        mutationFn: () => logsApi.clearLogs(),
        onSuccess: () => queryClient.invalidateQueries({ queryKey: ["logs"] }),
    });

    const scrollToBottom = useCallback(() => {
        if (autoScroll && bottomRef.current) {
            bottomRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [autoScroll]);

    useEffect(() => {
        scrollToBottom();
    }, [logsQ.data, scrollToBottom]);

    const logs = logsQ.data?.logs || [];
    const totalBuffered = logsQ.data?.total_buffered || 0;

    return (
        <div className="p-6 max-w-7xl mx-auto space-y-6">
            {/* Header */}
            <div className="flex justify-between items-end">
                <div>
                    <h1 className="text-3xl font-bold text-white tracking-tight flex items-center space-x-3">
                        <Terminal size={28} className="text-emerald-400" />
                        <span>System Logs</span>
                    </h1>
                    <p className="text-gray-400 mt-1">
                        Real-time application event stream &mdash;{" "}
                        <span className="text-emerald-400 font-semibold">{totalBuffered}</span> entries buffered
                    </p>
                </div>
                <div className="flex items-center space-x-3">
                    <button
                        onClick={() => setAutoRefresh(!autoRefresh)}
                        className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-bold uppercase tracking-wider transition-all border ${autoRefresh
                                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30 shadow-lg shadow-emerald-900/20"
                                : "bg-gray-800/50 text-gray-400 border-gray-700/50"
                            }`}
                    >
                        {autoRefresh ? <Pause size={14} /> : <Play size={14} />}
                        <span>{autoRefresh ? "Live" : "Paused"}</span>
                    </button>
                    <button
                        onClick={() => clearMut.mutate()}
                        disabled={clearMut.isPending}
                        className="flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-bold uppercase tracking-wider bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20 transition-all"
                    >
                        <Trash2 size={14} />
                        <span>Clear</span>
                    </button>
                </div>
            </div>

            {/* Filters */}
            <div className="flex flex-wrap items-center gap-3">
                {/* Search */}
                <div className="relative flex-1 min-w-[200px] max-w-md">
                    <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                    <input
                        type="text"
                        placeholder="Search logs..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="w-full bg-gray-900/60 border border-gray-700/50 rounded-xl pl-10 pr-4 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/20 transition-all"
                    />
                </div>

                {/* Level Filters */}
                <div className="flex items-center space-x-2">
                    <Filter size={14} className="text-gray-500" />
                    <button
                        onClick={() => setLevelFilter(null)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all border ${!levelFilter
                                ? "bg-blue-500/10 text-blue-400 border-blue-500/30"
                                : "bg-gray-800/30 text-gray-500 border-gray-700/30 hover:text-gray-300"
                            }`}
                    >
                        All
                    </button>
                    {Object.entries(LEVEL_CONFIG).map(([level, cfg]) => (
                        <button
                            key={level}
                            onClick={() => setLevelFilter(levelFilter === level ? null : level)}
                            className={`px-3 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all border ${levelFilter === level
                                    ? `${cfg.bg} ${cfg.color} ${cfg.border}`
                                    : "bg-gray-800/30 text-gray-500 border-gray-700/30 hover:text-gray-300"
                                }`}
                        >
                            {level}
                        </button>
                    ))}
                </div>

                {/* Auto scroll */}
                <button
                    onClick={() => setAutoScroll(!autoScroll)}
                    className={`flex items-center space-x-1 px-3 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all border ${autoScroll
                            ? "bg-indigo-500/10 text-indigo-400 border-indigo-500/30"
                            : "bg-gray-800/30 text-gray-500 border-gray-700/30"
                        }`}
                >
                    <ChevronDown size={14} />
                    <span>Auto-scroll</span>
                </button>
            </div>

            {/* Terminal Log Viewer */}
            <div className="bg-gray-950/80 border border-gray-700/50 rounded-2xl overflow-hidden shadow-2xl">
                {/* Terminal header bar */}
                <div className="flex items-center space-x-2 px-4 py-3 bg-gray-900/80 border-b border-gray-800/50">
                    <div className="w-3 h-3 rounded-full bg-red-500/80" />
                    <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                    <div className="w-3 h-3 rounded-full bg-green-500/80" />
                    <span className="ml-3 text-xs font-mono text-gray-500">
                        btc-oracle-logs — {logs.length} entries
                    </span>
                    {logsQ.isFetching && (
                        <RefreshCw size={12} className="ml-auto text-blue-400 animate-spin" />
                    )}
                </div>

                {/* Log entries */}
                <div className="max-h-[60vh] overflow-y-auto custom-scrollbar p-4 font-mono text-sm space-y-0.5">
                    {logs.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-20 text-gray-600">
                            <Terminal size={40} className="mb-3 opacity-20" />
                            <p className="text-sm italic">
                                {logsQ.isLoading ? "Loading logs..." : "No log entries found"}
                            </p>
                        </div>
                    ) : (
                        logs.map((log, i) => {
                            const cfg = LEVEL_CONFIG[log.level] || LEVEL_CONFIG.INFO;
                            const Icon = cfg.icon;
                            return (
                                <div
                                    key={i}
                                    className="flex items-start space-x-3 py-1.5 px-2 rounded-lg hover:bg-gray-800/30 transition-colors group"
                                >
                                    <span className="text-gray-600 text-[11px] font-mono min-w-[155px] shrink-0 mt-0.5">
                                        {log.timestamp}
                                    </span>
                                    <span
                                        className={`flex items-center space-x-1 min-w-[70px] shrink-0 text-[11px] font-bold uppercase tracking-wider ${cfg.color}`}
                                    >
                                        <Icon size={12} />
                                        <span>{log.level}</span>
                                    </span>
                                    <span className="text-gray-500 text-[11px] min-w-[80px] max-w-[120px] truncate shrink-0 mt-0.5" title={log.logger}>
                                        {log.logger}
                                    </span>
                                    <span className="text-gray-300 text-[13px] break-all leading-relaxed">
                                        {log.message}
                                    </span>
                                </div>
                            );
                        })
                    )}
                    <div ref={bottomRef} />
                </div>
            </div>
        </div>
    );
}
