const COLORS: Record<string, string> = {
  production: "bg-green-100 text-green-800",
  staging: "bg-amber-100 text-amber-800",
  completed: "bg-green-100 text-green-800",
  running: "bg-blue-100 text-blue-800",
  failed: "bg-red-100 text-red-800",
  skipped_insufficient_data: "bg-slate-200 text-slate-700",
  pending: "bg-slate-200 text-slate-700",
};

export default function Badge({ children }: { children: string }) {
  const color = COLORS[children] ?? "bg-slate-100 text-slate-700";
  return <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${color}`}>{children}</span>;
}
