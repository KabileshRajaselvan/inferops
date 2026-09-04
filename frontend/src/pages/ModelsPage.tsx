import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { listModels } from "../api/client";
import type { ModelSummary } from "../api/types";
import Badge from "../components/Badge";

export default function ModelsPage() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listModels()
      .then(setModels)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const names = Array.from(new Set(models.map((m) => m.name)));

  return (
    <div>
      <h1 className="text-xl font-semibold text-slate-900">Model Registry</h1>
      <p className="mt-1 text-sm text-slate-600">
        Every trained version, from the initial A/B pair through any automatic retrains.
      </p>

      {loading && <p className="mt-6 text-sm text-slate-500">Loading...</p>}
      {error && (
        <p className="mt-6 rounded-md bg-red-50 p-3 text-sm text-red-700">
          Could not reach the API ({error}). Is the stack running (`docker compose up`)?
        </p>
      )}

      {names.map((name) => (
        <div key={name} className="mt-6">
          <Link to={`/models/${encodeURIComponent(name)}`} className="text-lg font-medium text-slate-900 hover:underline">
            {name}
          </Link>
          <div className="mt-2 overflow-x-auto rounded-lg border border-slate-200 bg-white">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="px-4 py-2">Version</th>
                  <th className="px-4 py-2">Status</th>
                  <th className="px-4 py-2">Algorithm</th>
                  <th className="px-4 py-2">Accuracy</th>
                  <th className="px-4 py-2">ROC AUC</th>
                  <th className="px-4 py-2">Created</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {models
                  .filter((m) => m.name === name)
                  .map((m) => (
                    <tr key={m.id}>
                      <td className="px-4 py-2 font-medium">{m.version}</td>
                      <td className="px-4 py-2">
                        <Badge>{m.status}</Badge>
                      </td>
                      <td className="px-4 py-2 text-slate-600">{String(m.metrics.algorithm ?? "-")}</td>
                      <td className="px-4 py-2 text-slate-600">{String(m.metrics.accuracy ?? "-")}</td>
                      <td className="px-4 py-2 text-slate-600">{String(m.metrics.roc_auc ?? "-")}</td>
                      <td className="px-4 py-2 text-slate-500">{new Date(m.created_at).toLocaleString()}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}

      {!loading && !error && names.length === 0 && (
        <p className="mt-6 text-sm text-slate-500">
          No models registered yet. Run <code className="rounded bg-slate-100 px-1">python scripts/train_model.py</code> first.
        </p>
      )}
    </div>
  );
}
