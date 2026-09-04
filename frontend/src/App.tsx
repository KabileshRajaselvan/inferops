import { Link, Route, Routes } from "react-router-dom";

import ModelDetailPage from "./pages/ModelDetailPage";
import ModelsPage from "./pages/ModelsPage";

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-4">
          <Link to="/" className="text-lg font-semibold tracking-tight">
            InferOps
          </Link>
          <nav className="text-sm text-slate-600">
            <Link to="/" className="hover:text-slate-900">
              Models
            </Link>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">
        <Routes>
          <Route path="/" element={<ModelsPage />} />
          <Route path="/models/:name" element={<ModelDetailPage />} />
        </Routes>
      </main>
    </div>
  );
}
