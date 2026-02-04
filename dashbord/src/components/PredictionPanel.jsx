import React, { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { modelsApi } from "../services/api";

export default function PredictionPanel({ modelId, onResult }) {
  const [loading, setLoading] = useState(false);
  const mutation = useMutation({
    mutationFn: (payload) => modelsApi.predict(payload).then((r) => r.data),
  });

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await mutation.mutateAsync({ model_id: modelId });
      onResult && onResult(res);
    } catch (e) {
      onResult && onResult({ error: e?.message || "Prediction failed" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 bg-gray-800 rounded">
      <div className="flex items-center gap-3">
        <button
          onClick={handlePredict}
          disabled={loading}
          className="px-3 py-1 bg-orange-500 rounded text-white"
        >
          {loading ? "Predicting..." : "Predict with latest data"}
        </button>
      </div>
    </div>
  );
}
