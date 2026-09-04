import { useCallback, useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { evaluateModel, getModelMetrics, listDeployments, predict, updateDeployment } from "../api/client";
import type { DeploymentOut, ModelMetricsResponse, PredictResponse } from "../api/types";
import Badge from "../components/Badge";
import StatCard from "../components/StatCard";

const ITEM_CATEGORIES = ["electronics", "books", "fashion", "home", "sports", "beauty"] as const;
const DEVICES = ["mobile", "desktop", "tablet"] as const;

function randomFeatures() {
  return {
    user_age: Math.round(18 + Math.random() * 50),
    user_tenure_days: Math.round(Math.random() * 2000),
    user_avg_session_min: Math.round(Math.random() * 60 * 10) / 10,
    user_category_affinity: Math.round(Math.random() * 100) / 100,
    item_price: Math.round(Math.random() * 300 * 100) / 100,
    item_popularity: Math.round(Math.random() * 100) / 100,
    item_category: ITEM_CATEGORIES[Math.floor(Math.random() * ITEM_CATEGORIES.length)],
    hour_of_day: Math.floor(Math.random() * 24),
    history_click_rate: Math.round(Math.random() * 100) / 100,
    device: DEVICES[Math.floor(Math.random() * DEVICES.length)],
  };
}

export default function ModelDetailPage() {
  const { name } = useParams<{ name: string }>();
  const [metrics, setMetrics] = useState<ModelMetricsResponse | null>(null);
  const [deployments, setDeployments] = useState<DeploymentOut[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [tryItResult, setTryItResult] = useState<PredictResponse | null>(null);
  const [tryItBusy, setTryItBusy] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [evalMessage, setEvalMessage] = useState<string | null>(null);

  const load = useCallback(() => {
    if (!name) return;
    Promise.all([getModelMetrics(name, 7), listDeployments(name)])
      .then(([m, d]) => {
        setMetrics(m);
        setDeployments(d);
      })
      .catch((e) => setError(String(e)));
  }, [name]);

  useEffect(() => {
    load();
    const interval = setInterval(load, 15000);
    return () => clearInterval(interval);
  }, [load]);

  async function handleTraffic(id: string, value: number) {
    await updateDeployment(id, { traffic_split: value });
    load();
  }

  async function handleTryIt() {
    if (!name) return;
    setTryItBusy(true);
    try {
      const result = await predict(name, randomFeatures());
      setTryItResult(result);
    } catch (e) {
      setError(String(e));
    } finally {
      setTryItBusy(false);
    }
  }

  async function handleEvaluate() {
    if (!name) return;
    setEvaluating(true);
    setEvalMessage(null);
    try {
      const result = await evaluateModel(name);
      const triggered = result.versions.filter((v) => v.retrain_triggered);
      setEvalMessage(
        triggered.length > 0
          ? `Evaluation ran: retraining triggered for ${triggered.map((v) => `v${v.version}`).join(", ")}`
          : "Evaluation ran: no version crossed the accuracy-drop threshold",
      );
      load();
    } catch (e) {
      setError(String(e));
    } finally {
      setEvaluating(false);
    }
  }

  if (error) {
    return <p className="rounded-md bg-red-50 p-3 text-sm text-red-700">{error}</p>;
  }
  if (!metrics) {
    return <p className="text-sm text-slate-500">Loading...</p>;
  }

  const latest = metrics.daily[metrics.daily.length - 1];
  // metrics.drift is already ordered most-recent-first; keep only each feature's latest score
  // (the evaluator re-checks all features every pass, so naively slicing the raw list can show
  // the same feature twice from two different check cycles instead of eight distinct features).
  const seenFeatures = new Set<string>();
  const drift = metrics.drift
    .filter((d) => (seenFeatures.has(d.feature_name) ? false : (seenFeatures.add(d.feature_name), true)))
    .map((d) => ({ feature: d.feature_name, score: d.drift_score }));

  return (
    <div>
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-slate-900">{metrics.model_name}</h1>
        <button
          onClick={handleEvaluate}
          disabled={evaluating}
          className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
        >
          {evaluating ? "Evaluating..." : "Run evaluation now"}
        </button>
      </div>
      {evalMessage && <p className="mt-2 text-sm text-slate-600">{evalMessage}</p>}

      <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard label="Predictions (7d)" value={String(metrics.daily.reduce((s, d) => s + d.prediction_count, 0))} />
        <StatCard label="Click rate (latest day)" value={latest ? `${((latest.click_rate ?? 0) * 100).toFixed(1)}%` : "-"} />
        <StatCard label="Accuracy (latest day)" value={latest?.accuracy != null ? `${(latest.accuracy * 100).toFixed(1)}%` : "n/a"} hint={latest ? `${latest.labeled_count} labeled` : undefined} />
        <StatCard label="p95 latency (latest day)" value={latest?.p95_latency_ms != null ? `${latest.p95_latency_ms} ms` : "-"} />
      </div>

      <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <h2 className="text-sm font-medium text-slate-700">Accuracy & click rate</h2>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={metrics.daily}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="day" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="accuracy" stroke="#2563eb" name="accuracy" connectNulls />
              <Line type="monotone" dataKey="click_rate" stroke="#16a34a" name="click rate" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <h2 className="text-sm font-medium text-slate-700">Latency (ms)</h2>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={metrics.daily}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="day" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="avg_latency_ms" stroke="#9333ea" name="avg" />
              <Line type="monotone" dataKey="p95_latency_ms" stroke="#ea580c" name="p95" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <h2 className="text-sm font-medium text-slate-700">Data drift (KL-divergence, most recent checks)</h2>
          {drift.length === 0 ? (
            <p className="mt-4 text-sm text-slate-500">No drift checks recorded yet.</p>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={drift}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="feature" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="score" fill="#dc2626" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <h2 className="text-sm font-medium text-slate-700">A/B traffic split</h2>
          <div className="mt-4 space-y-4">
            {deployments.map((d) => (
              <div key={d.id}>
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">v{d.model_version}</span>
                  <span className="text-slate-500">{Math.round(d.traffic_split * 100)}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={d.traffic_split}
                  onChange={(e) => handleTraffic(d.id, Number(e.target.value))}
                  className="mt-1 w-full"
                />
              </div>
            ))}
            {deployments.length === 0 && <p className="text-sm text-slate-500">No production deployments yet.</p>}
          </div>
        </div>
      </div>

      <div className="mt-8 rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="text-sm font-medium text-slate-700">Retraining jobs</h2>
        {metrics.retraining_jobs.length === 0 ? (
          <p className="mt-2 text-sm text-slate-500">None triggered yet.</p>
        ) : (
          <table className="mt-2 min-w-full text-sm">
            <thead className="text-left text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="py-1 pr-4">Status</th>
                <th className="py-1 pr-4">Trigger</th>
                <th className="py-1 pr-4">New version</th>
                <th className="py-1 pr-4">Started</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {metrics.retraining_jobs.map((j) => (
                <tr key={j.id}>
                  <td className="py-1 pr-4">
                    <Badge>{j.status}</Badge>
                  </td>
                  <td className="py-1 pr-4 text-slate-600">{j.trigger_reason}</td>
                  <td className="py-1 pr-4 text-slate-600">{j.new_version ?? "-"}</td>
                  <td className="py-1 pr-4 text-slate-500">{j.started_at ? new Date(j.started_at).toLocaleString() : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="mt-8 rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="text-sm font-medium text-slate-700">Try it</h2>
        <p className="mt-1 text-sm text-slate-500">Sends one live POST /predict with random features through the A/B router.</p>
        <button
          onClick={handleTryIt}
          disabled={tryItBusy}
          className="mt-3 rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium hover:bg-slate-50 disabled:opacity-50"
        >
          {tryItBusy ? "Predicting..." : "Send a prediction"}
        </button>
        {tryItResult && (
          <div className="mt-3 rounded-md bg-slate-50 p-3 text-sm">
            <div>
              <span className="font-medium">{tryItResult.prediction.label}</span> (score{" "}
              {tryItResult.prediction.score.toFixed(3)}) - served by v{tryItResult.model_version}, deployment{" "}
              {tryItResult.deployment_id.slice(0, 8)}, {tryItResult.latency_ms}ms{tryItResult.cache_hit ? " (cache hit)" : ""}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
