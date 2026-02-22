import React from "react";
import { Routes, Route } from "react-router-dom";
import Header from "./components/layout/Header.jsx";
import Sidebar from "./components/layout/Sidebar.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Models from "./pages/Models.jsx";
import Comparison from "./pages/Comparison.jsx";
import Predictions from "./pages/Predictions.jsx";
import Backtesting from "./pages/Backtesting.jsx";
import Logs from "./pages/Logs.jsx";
import Settings from "./pages/Settings.jsx";
import "./App.css";

function App() {
  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <Header />
      <div className="flex">
        <Sidebar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/models" element={<Models />} />
            <Route path="/comparison" element={<Comparison />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/backtesting" element={<Backtesting />} />
            <Route path="/logs" element={<Logs />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

export default App;
