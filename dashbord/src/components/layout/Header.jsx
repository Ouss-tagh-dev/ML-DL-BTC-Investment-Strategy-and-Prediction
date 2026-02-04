import React from "react";
import { NavLink } from "react-router-dom";
import { LayoutDashboard, Cpu, LineChart, BrainCircuit, Wallet, Globe, Shield } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { systemApi } from "../../services/api";

export default function Header() {
  const [address, setAddress] = React.useState(null);
  const [isConnecting, setIsConnecting] = React.useState(false);

  const connectWallet = () => {
    setIsConnecting(true);
    setTimeout(() => {
      setAddress("0x71C7...f92A");
      setIsConnecting(false);
    }, 1200);
  };

  const healthQ = useQuery({
    queryKey: ["health"],
    queryFn: () => systemApi.getHealth().then((r) => r.data),
    refetchInterval: 30000,
    retry: false,
    refetchOnWindowFocus: false
  });

  return (
    <header className="sticky top-0 z-40 w-full bg-gray-950/50 backdrop-blur-xl border-b border-gray-800/50 px-6 py-4 flex items-center justify-between">
      <div className="flex items-center space-x-8">
        <NavLink to="/" className="flex items-center space-x-3 group">
          <div className="p-2 bg-gradient-to-tr from-blue-600 to-indigo-600 rounded-xl group-hover:rotate-12 transition-transform shadow-lg shadow-blue-900/40">
            <BrainCircuit className="text-white" size={24} />
          </div>
          <div className="flex flex-col">
            <span className="text-lg font-black bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400 leading-none">
              BTC ORACLE
            </span>
            <span className="text-[10px] font-bold text-blue-500 tracking-[0.2em] uppercase">
              Neural Engine v2.0
            </span>
          </div>
        </NavLink>

        <nav className="hidden xl:flex items-center space-x-1">
          <HeaderLink to="/" icon={<LayoutDashboard size={18} />} label="Overview" />
          <HeaderLink to="/models" icon={<Cpu size={18} />} label="Models" />
          <HeaderLink to="/comparison" icon={<LineChart size={18} />} label="Analytics" />
          <HeaderLink to="/predictions" icon={<BrainCircuit size={18} />} label="Inference" />
        </nav>
      </div>

      <div className="flex items-center space-x-6">
        <div className="hidden md:flex items-center space-x-4 bg-gray-900/50 px-4 py-2 rounded-2xl border border-gray-800/50">
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)] ${healthQ.isError ? "bg-red-500 shadow-red-500/50" :
                healthQ.isLoading ? "bg-yellow-500 shadow-yellow-500/50" : "bg-green-500 shadow-green-500/50"
              }`} />
            <span className="text-[10px] font-bold text-gray-400 uppercase tracking-tighter">
              {healthQ.isError ? "API Offline" : healthQ.isLoading ? "Connecting..." : "API Live Hub"}
            </span>
          </div>
          <div className="h-4 w-px bg-gray-800" />
          <div className="flex items-center space-x-2 text-xs text-gray-400 font-medium">
            <Globe size={14} className="text-blue-400" />
            <span>Binance API</span>
          </div>
        </div>

        <button
          onClick={connectWallet}
          className={`flex items-center space-x-2 px-5 py-2.5 rounded-xl transition-all shadow-lg active:scale-95 border ${address
            ? "bg-green-500/10 text-green-400 border-green-500/20"
            : "bg-blue-600 hover:bg-blue-500 text-white border-blue-400/20 shadow-blue-900/20"
            }`}
        >
          {isConnecting ? (
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
          ) : address ? (
            <Shield size={16} />
          ) : (
            <Wallet size={16} />
          )}
          <span className="text-xs font-bold uppercase tracking-tight">
            {isConnecting ? "Validating..." : address || "Connect Wallet"}
          </span>
        </button>
      </div>
    </header>
  );
}

function HeaderLink({ to, icon, label }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex items-center space-x-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${isActive
          ? "bg-blue-600/10 text-blue-400 border border-blue-500/20"
          : "text-gray-400 hover:text-white hover:bg-gray-800/40"
        }`
      }
    >
      {icon}
      <span>{label}</span>
    </NavLink>
  );
}
