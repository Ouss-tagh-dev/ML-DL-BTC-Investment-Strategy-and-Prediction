import React, { useState } from "react";
import { useModels } from "../hooks/useApi";
import ModelCard from "../components/ModelCard";
import ModelDetailsModal from "../components/ModelDetailsModal";
import PredictionPanel from "../components/PredictionPanel";

export default function Models() {
  const { data: models = [], isLoading } = useModels();
  const [selected, setSelected] = useState(null);
  const [result, setResult] = useState(null);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Models</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {isLoading && <div>Loading models...</div>}
        {models?.map((m) => (
          <ModelCard
            key={m.id}
            model={m}
            onPredict={(id) => setSelected({ predictId: id })}
            onViewDetails={(id) => setSelected({ detailsId: id })}
          />
        ))}
      </div>

      {selected?.detailsId && (
        <ModelDetailsModal
          modelId={selected.detailsId}
          onClose={() => setSelected(null)}
        />
      )}

      {selected?.predictId && (
        <div className="mt-6">
          <h2 className="text-lg font-semibold mb-2">Live Prediction</h2>
          <PredictionPanel
            modelId={selected.predictId}
            onResult={(r) => setResult(r)}
          />
          {result && (
            <pre className="mt-4 bg-gray-800 p-3 rounded">
              {JSON.stringify(result, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
