import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { settingsApi } from "../services/api";
import {
    Settings as SettingsIcon,
    Server,
    Cpu,
    Database,
    Shield,
    Activity,
    Check,
    AlertCircle,
    MemoryStick,
    Clock,
    Globe,
    ToggleLeft,
    ToggleRight,
} from "lucide-react";

export default function Settings() {
    const queryClient = useQueryClient();
    const [toast, setToast] = useState(null);

    const settingsQ = useQuery({
        queryKey: ["settings"],
        queryFn: () => settingsApi.getSettings().then((r) => r.data),
    });

    const sysInfoQ = useQuery({
        queryKey: ["system-info"],
        queryFn: () => settingsApi.getSystemInfo().then((r) => r.data),
        refetchInterval: 10000,
    });

    const updateMut = useMutation({
        mutationFn: (payload) => settingsApi.updateSettings(payload),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["settings"] });
            queryClient.invalidateQueries({ queryKey: ["system-info"] });
            showToast("Settings saved successfully", "success");
        },
        onError: (err) => {
            showToast(err.response?.data?.detail || "Failed to save", "error");
        },
    });

    const showToast = (message, type) => {
        setToast({ message, type });
        setTimeout(() => setToast(null), 3000);
    };

    const handleToggle = (field, currentValue) => {
        updateMut.mutate({ [field]: !currentValue });
    };

    const handleLogLevel = (level) => {
        updateMut.mutate({ log_level: level });
    };

    const cfg = settingsQ.data || {};
    const sys = sysInfoQ.data || {};

    return (
        <div className="p-6 max-w-7xl mx-auto space-y-6">
            {/* Toast */}
            {toast && (
                <div
                    className={`fixed top-6 right-6 z-50 flex items-center space-x-2 px-5 py-3 rounded-xl shadow-2xl border text-sm font-bold animate-pulse ${toast.type === "success"
                        ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                        : "bg-red-500/10 text-red-400 border-red-500/30"
                        }`}
                >
                    {toast.type === "success" ? <Check size={16} /> : <AlertCircle size={16} />}
                    <span>{toast.message}</span>
                </div>
            )}

            {/* Header */}
            <div>
                <h1 className="text-3xl font-bold text-white tracking-tight flex items-center space-x-3">
                    <SettingsIcon size={28} className="text-blue-400" />
                    <span>Settings</span>
                </h1>
                <p className="text-gray-400 mt-1">Configure your BTC Oracle dashboard and system parameters</p>
            </div>

            {/* System Info Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <SysCard
                    icon={<Server size={20} className="text-blue-400" />}
                    label="Python"
                    value={sys.python_version || "—"}
                    sub={sys.platform || ""}
                />
                <SysCard
                    icon={<Clock size={20} className="text-emerald-400" />}
                    label="Uptime"
                    value={sys.uptime || "—"}
                    sub={`Started ${sys.started_at ? new Date(sys.started_at).toLocaleTimeString() : "—"}`}
                />
                <SysCard
                    icon={<MemoryStick size={20} className="text-purple-400" />}
                    label="Memory"
                    value={sys.memory ? `${sys.memory.used_gb}/${sys.memory.total_gb} GB` : "N/A"}
                    sub={sys.memory ? `${sys.memory.percent}% used` : "psutil not installed"}
                />
                <SysCard
                    icon={<Cpu size={20} className="text-amber-400" />}
                    label="CPU"
                    value={sys.cpu_percent != null ? `${sys.cpu_percent}%` : "N/A"}
                    sub={sys.architecture || "—"}
                />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* API Configuration */}
                <SettingsSection
                    icon={<Globe size={20} className="text-blue-400" />}
                    title="API Configuration"
                >
                    <ReadOnlyField label="Project Name" value={cfg.api?.project_name} />
                    <ReadOnlyField label="Version" value={cfg.api?.version} />
                    <ReadOnlyField label="API Prefix" value={cfg.api?.prefix} />
                    <ReadOnlyField label="Description" value={cfg.api?.description} />
                </SettingsSection>

                {/* CORS Settings */}
                <SettingsSection
                    icon={<Shield size={20} className="text-emerald-400" />}
                    title="CORS Origins"
                >
                    <div className="space-y-2">
                        {(cfg.cors?.origins || []).map((origin, i) => (
                            <div
                                key={i}
                                className="flex items-center space-x-2 bg-gray-900/50 px-4 py-2.5 rounded-xl border border-gray-700/30"
                            >
                                <Globe size={14} className="text-gray-500" />
                                <span className="text-sm text-gray-300 font-mono">{origin}</span>
                            </div>
                        ))}
                    </div>
                </SettingsSection>

                {/* Model Management */}
                <SettingsSection
                    icon={<Cpu size={20} className="text-purple-400" />}
                    title="Model Management"
                >
                    <ToggleField
                        label="Cache Models in Memory"
                        description="Keep loaded models in RAM for instant inference"
                        enabled={cfg.models?.cache_enabled ?? true}
                        onToggle={() => handleToggle("cache_models", cfg.models?.cache_enabled)}
                        loading={updateMut.isPending}
                    />
                    <div className="mt-4">
                        <p className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3">
                            Available Models ({(cfg.models?.available || []).length})
                        </p>
                        <div className="grid grid-cols-2 gap-2">
                            {(cfg.models?.available || []).map((model) => (
                                <div
                                    key={model}
                                    className="flex items-center space-x-2 bg-gray-900/50 px-3 py-2 rounded-xl border border-gray-700/30"
                                >
                                    <Activity size={12} className="text-blue-400" />
                                    <span className="text-xs text-gray-300 font-medium">{model.replace(/_/g, " ")}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </SettingsSection>

                {/* Data & Cache */}
                <SettingsSection
                    icon={<Database size={20} className="text-amber-400" />}
                    title="Data & Cache"
                >
                    <ToggleField
                        label="Cache Data in Memory"
                        description="Keep historical data in RAM for faster queries"
                        enabled={cfg.data?.cache_enabled ?? true}
                        onToggle={() => handleToggle("cache_data", cfg.data?.cache_enabled)}
                        loading={updateMut.isPending}
                    />
                    <div className="mt-4 space-y-2">
                        <ReadOnlyField label="Data Directory" value={cfg.data?.data_dir} mono />
                        <ReadOnlyField label="Models Directory" value={cfg.data?.models_dir} mono />
                    </div>
                </SettingsSection>

                {/* Logging Configuration */}
                <SettingsSection
                    icon={<Activity size={20} className="text-red-400" />}
                    title="Logging"
                    fullWidth
                >
                    <p className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3">Log Level</p>
                    <div className="flex flex-wrap gap-2">
                        {["DEBUG", "INFO", "WARNING", "ERROR"].map((level) => {
                            const isActive = sys.log_level === level;
                            const colors = {
                                DEBUG: "bg-sky-500/10 text-sky-400 border-sky-500/30",
                                INFO: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
                                WARNING: "bg-amber-500/10 text-amber-400 border-amber-500/30",
                                ERROR: "bg-red-500/10 text-red-400 border-red-500/30",
                            };
                            return (
                                <button
                                    key={level}
                                    onClick={() => handleLogLevel(level)}
                                    disabled={updateMut.isPending}
                                    className={`px-4 py-2 rounded-xl text-xs font-bold uppercase tracking-wider transition-all border ${isActive
                                        ? `${colors[level]} shadow-lg`
                                        : "bg-gray-800/30 text-gray-500 border-gray-700/30 hover:text-gray-300 hover:border-gray-600"
                                        }`}
                                >
                                    {level}
                                </button>
                            );
                        })}
                    </div>
                    <p className="mt-2 text-xs text-gray-600">
                        Current level: <span className="text-gray-400 font-semibold">{sys.log_level || "—"}</span>
                    </p>
                </SettingsSection>
            </div>
        </div>
    );
}

function SysCard({ icon, label, value, sub }) {
    return (
        <div className="bg-gray-800/40 border border-gray-700/50 backdrop-blur-sm p-5 rounded-2xl shadow-lg group hover:bg-gray-800/60 transition-all">
            <div className="flex justify-between items-start mb-2">
                <span className="text-gray-400 text-sm font-medium">{label}</span>
                <div className="p-2 bg-gray-900/50 rounded-xl group-hover:scale-110 transition-transform">
                    {icon}
                </div>
            </div>
            <div className="text-xl font-bold text-white mb-1 truncate" title={value}>
                {value}
            </div>
            <div className="text-xs text-gray-500 truncate" title={sub}>
                {sub}
            </div>
        </div>
    );
}

function SettingsSection({ icon, title, children, fullWidth }) {
    return (
        <div
            className={`bg-gray-800/40 border border-gray-700/50 p-6 rounded-2xl space-y-4 ${fullWidth ? "lg:col-span-2" : ""
                }`}
        >
            <h3 className="font-semibold text-lg flex items-center space-x-2">
                {icon}
                <span>{title}</span>
            </h3>
            {children}
        </div>
    );
}

function ReadOnlyField({ label, value, mono }) {
    return (
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between bg-gray-900/40 px-4 py-3 rounded-xl border border-gray-700/20">
            <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">{label}</span>
            <span
                className={`text-sm text-gray-300 mt-1 sm:mt-0 truncate max-w-[300px] ${mono ? "font-mono text-xs" : ""
                    }`}
                title={value}
            >
                {value || "—"}
            </span>
        </div>
    );
}

function ToggleField({ label, description, enabled, onToggle, loading }) {
    return (
        <div className="flex items-center justify-between bg-gray-900/40 px-4 py-3 rounded-xl border border-gray-700/20">
            <div>
                <p className="text-sm font-semibold text-white">{label}</p>
                <p className="text-xs text-gray-500">{description}</p>
            </div>
            <button
                onClick={onToggle}
                disabled={loading}
                className={`transition-all ${enabled ? "text-emerald-400" : "text-gray-600"}`}
            >
                {enabled ? <ToggleRight size={32} /> : <ToggleLeft size={32} />}
            </button>
        </div>
    );
}
