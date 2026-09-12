import type { RunView } from "@/lib/act/engine";
import { formatCompact, formatLoss } from "@/lib/utils";

export function MetricsStrip({ run }: { run: RunView }) {
  const items = [
    { k: "Step", v: `${run.step.toLocaleString()} / ${run.maxSteps.toLocaleString()}` },
    { k: "Tokens", v: formatCompact(run.tokens, 1) },
    { k: "Val loss", v: formatLoss(run.valLoss) },
    { k: "Quality", v: run.quality.toFixed(3) },
    { k: "Params", v: formatCompact(run.params, 1) },
    { k: "Active", v: formatCompact(run.activeParams, 1) },
    { k: "FLOPs/tok", v: formatCompact(run.flopsPerToken, 1) },
    { k: "Mutations", v: String(run.mutationCount) },
  ];
  return (
    <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-lg bg-border sm:grid-cols-4 lg:grid-cols-8">
      {items.map((it) => (
        <div key={it.k} className="bg-surface px-3 py-3">
          <dt className="font-mono text-[0.625rem] uppercase tracking-[0.14em] text-muted">
            {it.k}
          </dt>
          <dd className="mt-1 font-mono text-sm tabular text-fg">{it.v}</dd>
        </div>
      ))}
    </dl>
  );
}
