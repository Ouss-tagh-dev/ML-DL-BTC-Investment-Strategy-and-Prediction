import { useQuery, useMutation } from "@tanstack/react-query";
import { modelsApi, dataApi } from "../services/api";

export const useModels = (options = {}) => {
  return useQuery({
    queryKey: ["models"],
    queryFn: () => modelsApi.list().then((r) => r.data),
    ...options,
  });
};

export const usePredict = () => {
  return useMutation({
    mutationFn: (payload) => modelsApi.predict(payload).then((r) => r.data),
  });
};

export const useBatchPredict = () => {
  return useMutation({
    mutationFn: (payload) => modelsApi.batchPredict(payload).then((r) => r.data),
  });
};

export const useHistoricalData = (params, options = {}) => {
  return useQuery({
    queryKey: ["historical", params],
    queryFn: () => dataApi.getHistorical(params).then((r) => r.data),
    ...options,
  });
};

export const useLatestData = (n = 1, options = {}) => {
  return useQuery({
    queryKey: ["latest", n],
    queryFn: () => dataApi.getLatest(n).then((r) => r.data),
    ...options,
  });
};
