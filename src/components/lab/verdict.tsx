import type { RunView } from "@/lib/act/engine";
import { formatLoss } from "@/lib/utils";

export function Verdict({ runs }: { runs: RunView[] }) {
  const a = runs.find((r) => r.mode === "frozen");
  const b = runs.find((r) => r.mode === "random");
  const c = runs.find((r) => r.mode === "adaptive");
  if (!a || !b || !c) return null;
  const done = runs.every((r) => r.status === "done");
  if (!done) return null;

  const delta = c.quality - b.quality;
  const vsFixed = c.quality - a.quality;
  let title = "Inconclusive";
  let body =
    "Adaptive and random churn landed within noise. Gate 3 is not passed — simplify the controller before adding mechanisms.";
  if (delta > 0.02) {
    title = "Gate 3 · Adaptive > Random";
    body = `Information-guided topology beat identical random churn by ${(delta * 100).toFixed(1)} quality points. The importance system is earning its keep.`;
  } else if (delta < -0.02) {
    title = "Adaptive lost to churn";
    body =
      "Random topology movement beat the controller. Treat this as a failed trial — inspect mutation logs before adding Fisher or ablation.";
  }

  return (
    <aside className="rounded-lg bg-surface p-4 shadow-[var(--shadow-border)]">
      <p className="font-mono text-[0.6875rem] uppercase tracking-[0.18em] text-muted">
        Experiment 0 · result
      </p>
      <h2 className="mt-1 text-lg font-medium">{title}</h2>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted">{body}</p>
      <dl className="mt-4 grid grid-cols-3 gap-3 text-sm">
        <Stat label="A Fixed" loss={a.valLoss} q={a.quality} />
        <Stat label="B Random" loss={b.valLoss} q={b.quality} />
        <Stat label="C Adaptive" loss={c.valLoss} q={c.quality} />
      </dl>
      <p className="mt-3 font-mono text-[0.6875rem] text-subtle">
        ΔC−B quality {delta >= 0 ? "+" : ""}
        {delta.toFixed(3)} · ΔC−A {vsFixed >= 0 ? "+" : ""}
        {vsFixed.toFixed(3)} · rollbacks C {c.rollbackCount} / B {b.rollbackCount}
      </p>
    </aside>
  );
}

function Stat({ label, loss, q }: { label: string; loss: number; q: number }) {
  return (
    <div>
      <dt className="font-mono text-[0.625rem] uppercase tracking-wider text-muted">
        {label}
      </dt>
      <dd className="mt-0.5 font-mono tabular">
        {formatLoss(loss)}{" "}
        <span className="text-muted">Q {q.toFixed(3)}</span>
      </dd>
    </div>
  );
}
