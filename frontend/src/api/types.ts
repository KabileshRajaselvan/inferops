export interface ModelSummary {
  id: string;
  name: string;
  version: string;
  model_type: string;
  framework: string;
  status: string;
  metrics: Record<string, number | string>;
  mlflow_run_id: string | null;
  created_at: string;
}

export interface DailyMetric {
  day: string;
  prediction_count: number;
  avg_latency_ms: number | null;
  p95_latency_ms: number | null;
  click_rate: number | null;
  accuracy: number | null;
  labeled_count: number;
}

export interface DriftPoint {
  feature_name: string;
  drift_score: number;
  checked_at: string;
}

export interface DeploymentInfo {
  id: string;
  model_version: string;
  traffic_split: number;
  is_active: boolean;
}

export interface RetrainingJobInfo {
  id: string;
  trigger_reason: string;
  status: string;
  new_version: string | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface ModelMetricsResponse {
  model_name: string;
  start_date: string;
  end_date: string;
  daily: DailyMetric[];
  drift: DriftPoint[];
  deployments: DeploymentInfo[];
  retraining_jobs: RetrainingJobInfo[];
}

export interface DeploymentOut {
  id: string;
  model_id: string;
  model_name: string;
  model_version: string;
  environment: string;
  traffic_split: number;
  is_active: boolean;
  deployed_at: string;
}

export interface PredictFeatures {
  user_age: number;
  user_tenure_days: number;
  user_avg_session_min: number;
  user_category_affinity: number;
  item_price: number;
  item_popularity: number;
  item_category: "electronics" | "books" | "fashion" | "home" | "sports" | "beauty";
  hour_of_day: number;
  history_click_rate: number;
  device: "mobile" | "desktop" | "tablet";
}

export interface PredictResponse {
  inference_id: string;
  prediction: { score: number; label: "click" | "no_click" };
  model_version: string;
  latency_ms: number;
  deployment_id: string;
  cache_hit: boolean;
}

export interface EvaluateVersionResult {
  version: string;
  model_id: string;
  drift: Record<string, number>;
  live_accuracy: number | null;
  labeled_count: number;
  baseline_accuracy: number | null;
  retrain_triggered: boolean;
  retraining_job?: { id: string; status: string };
}

export interface EvaluateResult {
  model_name: string;
  versions: EvaluateVersionResult[];
}
