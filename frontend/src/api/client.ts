import type {
  DeploymentOut,
  EvaluateResult,
  ModelMetricsResponse,
  ModelSummary,
  PredictFeatures,
  PredictResponse,
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8010";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }
  return response.json() as Promise<T>;
}

export function listModels(): Promise<ModelSummary[]> {
  return request("/api/v1/models");
}

export function getModelVersions(name: string): Promise<ModelSummary[]> {
  return request(`/api/v1/models/${encodeURIComponent(name)}`);
}

export function getModelMetrics(name: string, days = 7): Promise<ModelMetricsResponse> {
  const end = new Date();
  const start = new Date(end.getTime() - days * 24 * 60 * 60 * 1000);
  const params = new URLSearchParams({
    start_date: start.toISOString().slice(0, 10),
    end_date: end.toISOString().slice(0, 10),
  });
  return request(`/api/v1/models/${encodeURIComponent(name)}/metrics?${params}`);
}

export function evaluateModel(name: string): Promise<EvaluateResult> {
  return request(`/api/v1/models/${encodeURIComponent(name)}/evaluate`, { method: "POST" });
}

export function listDeployments(modelName?: string): Promise<DeploymentOut[]> {
  const params = modelName ? `?model_name=${encodeURIComponent(modelName)}` : "";
  return request(`/api/v1/deployments${params}`);
}

export function updateDeployment(
  id: string,
  patch: { traffic_split?: number; is_active?: boolean },
): Promise<DeploymentOut> {
  return request(`/api/v1/deployments/${id}`, { method: "PATCH", body: JSON.stringify(patch) });
}

export function predict(modelName: string, features: PredictFeatures): Promise<PredictResponse> {
  return request("/api/v1/predict", {
    method: "POST",
    body: JSON.stringify({ model_name: modelName, features }),
  });
}
