import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

export default function PriceChart({ data = [], height = 300 }) {
  return (
    <div style={{ width: "100%", height }} className="bg-gray-800 p-3 rounded">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="time" stroke="#cbd5e1" />
          <YAxis stroke="#cbd5e1" />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #4b5563",
            }}
          />
          <Legend />
          <Line type="monotone" dataKey="close" stroke="#60a5fa" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
