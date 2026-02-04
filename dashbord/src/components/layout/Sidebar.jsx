import React from "react";
import { NavLink } from "react-router-dom";
import { LayoutDashboard, Cpu, LineChart, BrainCircuit, History, Info, Settings } from "lucide-react";

export default function Sidebar() {
  return (
    <aside className="w-72 bg-gray-950/20 backdrop-blur-md border-r border-gray-800/50 p-6 hidden lg:flex flex-col h-[calc(100vh-80px)] sticky top-20">
      <div className="flex-1 space-y-8">
        <div>
          <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-[0.2em] mb-4 px-4">
            Main Navigation
          </h4>
          <ul className="space-y-1.5">
            <SidebarLink to="/" icon={<LayoutDashboard size={18} />} label="Overview" />
            <SidebarLink to="/models" icon={<Cpu size={18} />} label="Models" />
            <SidebarLink to="/comparison" icon={<LineChart size={18} />} label="Analytics" />
            <SidebarLink to="/predictions" icon={<BrainCircuit size={18} />} label="Inference Hub" />
          </ul>
        </div>

        <div>
          <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-[0.2em] mb-4 px-4">
            Analysis
          </h4>
          <ul className="space-y-1.5">
            <SidebarLink to="/backtesting" icon={<History size={18} />} label="Backtesting" />
            <SidebarLink to="/logs" icon={<Info size={18} />} label="System Logs" />
          </ul>
        </div>
      </div>

      <div className="pt-6 border-t border-gray-800/50">
        <SidebarLink to="/settings" icon={<Settings size={18} />} label="Settings" />
      </div>
    </aside>
  );
}

function SidebarLink({ to, icon, label }) {
  return (
    <li>
      <NavLink
        to={to}
        className={({ isActive }) =>
          `flex items-center space-x-3 px-4 py-3 rounded-2xl text-sm font-semibold transition-all group ${isActive
            ? "bg-blue-600 text-white shadow-lg shadow-blue-900/40"
            : "text-gray-400 hover:text-white hover:bg-gray-800/40"
          }`
        }
      >
        <div className="group-hover:scale-110 transition-transform">
          {icon}
        </div>
        <span>{label}</span>
      </NavLink>
    </li>
  );
}
