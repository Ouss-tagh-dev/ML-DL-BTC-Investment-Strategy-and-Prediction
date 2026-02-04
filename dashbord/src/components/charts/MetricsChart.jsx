import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";

export default function MetricsChart({
  data = [],
  metric = "accuracy",
  height = 300,
}) {
  return (
    <div style={{ width: "100%", height }} className="bg-gray-800 p-3 rounded">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <XAxis dataKey="name" stroke="#cbd5e1" />
          <YAxis stroke="#cbd5e1" />
          <Tooltip />
          <Legend />
          <Bar dataKey={metric} fill="#60a5fa" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
