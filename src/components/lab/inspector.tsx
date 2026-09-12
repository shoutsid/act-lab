import { useState } from "react";
import type { RunView } from "@/lib/act/engine";
import { serializeGenome, TASKS, type Lifecycle } from "@/lib/act/genome";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Download } from "lucide-react";

const TABS = ["Genome", "Importance", "Experts", "Mutations"] as const;

const LIFE_TONE: Record<Lifecycle, "muted" | "good" | "warn" | "moe" | "att"> = {
  new: "att",
  emerging: "att",
  active: "muted",
  stable: "good",
  essential: "good",
  uncertain: "warn",
  redundant: "moe",
  dying: "warn",
};

export function Inspector({ run }: { run: RunView }) {
  const [tab, setTab] = useState<(typeof TABS)[number]>("Genome");
  return (
    <section className="flex min-h-[28rem] flex-col rounded-lg bg-surface shadow-[var(--shadow-border)]">
      <div className="flex gap-1 overflow-x-auto border-b border-border px-2 pt-2">
        {TABS.map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={cn(
              "h-10 shrink-0 rounded-t-sm px-3 text-sm transition-colors duration-150",
              tab === t ? "bg-surface-2 text-fg" : "text-muted hover:text-fg",
            )}
          >
            {t}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-3 sm:p-4">
        {tab === "Genome" ? <GenomeTab run={run} /> : null}
        {tab === "Importance" ? <ImportanceTab run={run} /> : null}
        {tab === "Experts" ? <ExpertsTab run={run} /> : null}
        {tab === "Mutations" ? <MutationsTab run={run} /> : null}
      </div>
    </section>
  );
}

function GenomeTab({ run }: { run: RunView }) {
  const json = serializeGenome(run.genome);
  const download = () => {
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `genome-${run.mode}-${run.seed}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <p className="font-mono text-[0.6875rem] text-muted">
          hash {run.genomeHash.slice(0, 12)} · depth {run.genome.depth} · hidden{" "}
          {run.genome.hiddenSize}
        </p>
        <Button size="sm" variant="outline" onClick={download}>
          <Download />
          genome.json
        </Button>
      </div>
      <ul className="space-y-1">
        {run.genome.layers.map((l) => (
          <li
            key={l.id}
            className={cn(
              "flex items-center gap-2 rounded-sm px-2 py-1.5 font-mono text-xs",
              l.enabled ? "bg-surface-2/60" : "opacity-40",
            )}
          >
            <span className="w-8 text-subtle">L{String(l.index).padStart(2, "0")}</span>
            <Badge tone={l.module === "attention" ? "att" : l.module === "gdn" ? "gdn" : "muted"}>
              {l.module}
            </Badge>
            <Badge tone={l.ffn === "moe" ? "moe" : "muted"}>{l.ffn}</Badge>
            <span className="ml-auto text-muted">
              {l.ffn === "moe"
                ? `${l.experts.length}e / ${l.activeExperts}k`
                : `ffn ${l.ffnWidth}`}
              {l.module !== "gdn" ? ` · h${l.heads}` : ""}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ImportanceTab({ run }: { run: RunView }) {
  const rows = run.importance
    .filter((r) => r.kind === "expert")
    .slice()
    .sort((a, b) => b.activation - a.activation)
    .slice(0, 24);
  if (!rows.length) {
    return <p className="text-sm text-muted">No expert importance yet — wait for the stats clock.</p>;
  }
  return (
    <table className="w-full text-left text-xs">
      <thead className="font-mono text-[0.625rem] uppercase tracking-wider text-muted">
        <tr>
          <th className="py-1 font-medium">Component</th>
          <th className="py-1 font-medium">Act</th>
          <th className="py-1 font-medium">Fish</th>
          <th className="py-1 font-medium">Uniq</th>
          <th className="py-1 font-medium">Stab</th>
          <th className="py-1 font-medium">Life</th>
        </tr>
      </thead>
      <tbody className="font-mono tabular">
        {rows.map((r) => (
          <tr key={r.id} className="border-t border-border/80">
            <td className="py-1.5 text-fg">{r.label}</td>
            <td>{r.activation.toFixed(2)}</td>
            <td>{r.fisher.toFixed(2)}</td>
            <td>{r.uniqueness.toFixed(2)}</td>
            <td>{r.stability.toFixed(2)}</td>
            <td>
              <Badge tone={LIFE_TONE[r.lifecycle]}>{r.lifecycle}</Badge>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ExpertsTab({ run }: { run: RunView }) {
  if (!run.experts.length) {
    return <p className="text-sm text-muted">Dense layers have no expert ecology.</p>;
  }
  return (
    <ul className="grid gap-2 sm:grid-cols-2">
      {run.experts.slice(0, 32).map((e) => (
        <li key={e.id} className="rounded-sm bg-surface-2 p-2.5">
          <div className="flex items-center justify-between gap-2">
            <span className="font-mono text-[0.6875rem] text-muted">
              L{String(e.layerIndex).padStart(2, "0")} · {e.id.slice(0, 7)}
            </span>
            <Badge tone={LIFE_TONE[e.lifecycle]}>{e.lifecycle}</Badge>
          </div>
          <div className="mt-2 flex h-8 items-end gap-0.5">
            {e.specialty.map((s, i) => (
              <div
                key={TASKS[i]}
                title={`${TASKS[i]} ${(s * 100).toFixed(0)}`}
                className="flex-1 rounded-[2px] bg-accent/80"
                style={{ height: `${Math.max(8, s * 100)}%` }}
              />
            ))}
          </div>
          <p className="mt-1.5 font-mono text-[0.625rem] text-muted">
            route {e.routing.toFixed(2)} · uniq {e.uniqueness.toFixed(2)}
          </p>
        </li>
      ))}
    </ul>
  );
}

function MutationsTab({ run }: { run: RunView }) {
  if (!run.mutations.length) {
    return <p className="text-sm text-muted">No structural events yet.</p>;
  }
  return (
    <ol className="space-y-2">
      {[...run.mutations].reverse().map((m, i) => (
        <li
          key={`${m.step}-${m.op}-${i}`}
          className="rounded-sm bg-surface-2 px-2.5 py-2"
        >
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={m.op === "rollback" ? "warn" : m.accepted ? "good" : "muted"}>
              {m.op}
            </Badge>
            <span className="font-mono text-[0.6875rem] text-muted">
              step {m.step.toLocaleString()}
            </span>
            {!m.accepted ? (
              <Badge tone="warn">rejected</Badge>
            ) : null}
          </div>
          <p className="mt-1 text-xs leading-relaxed text-fg">{m.reason}</p>
          <p className="mt-0.5 font-mono text-[0.625rem] text-subtle">{m.target}</p>
        </li>
      ))}
    </ol>
  );
}
