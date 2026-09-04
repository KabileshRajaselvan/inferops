from prometheus_client import Counter, Gauge, Histogram

predictions_total = Counter(
    "inferops_predictions_total", "Total predictions served", ["model_name", "model_version"]
)
prediction_latency_ms = Histogram(
    "inferops_prediction_latency_ms",
    "Prediction latency in milliseconds",
    ["model_name"],
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500),
)
cache_hits_total = Counter("inferops_cache_hits_total", "Cache hits")
cache_misses_total = Counter("inferops_cache_misses_total", "Cache misses")
model_accuracy = Gauge("inferops_model_accuracy", "Rolling accuracy from labeled feedback", ["model_name", "version"])
data_drift_score = Gauge("inferops_data_drift_score", "Latest KL-divergence drift score", ["model_name", "feature"])
retraining_jobs_total = Counter("inferops_retraining_jobs_total", "Retraining jobs triggered", ["trigger_reason"])
batch_predict_rows_total = Counter("inferops_batch_predict_rows_total", "Rows processed via batch-predict")
