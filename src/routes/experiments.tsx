import { createFileRoute, Link } from "@tanstack/react-router";
import { AppShell } from "@/components/layout/shell";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useLab } from "@/lib/act/store";
import { formatCompact, formatLoss } from "@/lib/utils";
import { LossChart, ParetoChart } from "@/components/lab/charts";
import type { RunView } from "@/lib/act/engine";
import { useEffect, useMemo } from "react";

export const Route = createFileRoute("/experiments")({ component: ExperimentsPage });

function ExperimentsPage() {
  const saved = useLab((s) => s.saved);
  const compareIds = useLab((s) => s.compareIds);
  const toggleCompare = useLab((s) => s.toggleCompare);
  const loadSaved = useLab((s) => s.loadSaved);

  useEffect(() => {
    loadSaved();
  }, [loadSaved]);

  const compared: RunView[] = useMemo(() => {
    const out: RunView[] = [];
    for (const exp of saved) {
      if (!compareIds.includes(exp.id)) continue;
      const best =
        exp.views.find((v) => v.mode === "adaptive") ??
        exp.views[exp.views.length - 1];
      if (best) out.push({ ...best, id: exp.id, name: label(exp) });
    }
    return out;
  }, [saved, compareIds]);

  return (
    <AppShell>
      <main className="px-4 py-8 sm:px-6">
        <p className="font-mono text-[0.6875rem] uppercase tracking-[0.18em] text-muted">
          Scientific ledger
        </p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">Experiments</h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
          Runs saved from the lab persist in this browser. Select up to three to
          overlay quality and FLOPs. No result is considered complete without
          genome hash, seed, and controller mode.
        </p>

        {saved.length === 0 ? (
          <div className="mt-10 rounded-xl bg-surface px-5 py-12 text-center shadow-[var(--shadow-border)]">
            <p className="text-sm text-muted">
              The ledger is empty. Train in the lab, then save a run.
            </p>
            <Button asChild className="mt-4">
              <Link to="/">Open the lab</Link>
            </Button>
          </div>
        ) : (
          <>
            <div className="mt-6 overflow-x-auto rounded-lg bg-surface shadow-[var(--shadow-border)]">
              <table className="w-full min-w-[44rem] text-left text-sm">
                <thead className="font-mono text-[0.625rem] uppercase tracking-wider text-muted">
                  <tr className="border-b border-border">
                    <th className="px-3 py-3 font-medium">Compare</th>
                    <th className="px-3 py-3 font-medium">When</th>
                    <th className="px-3 py-3 font-medium">Hypothesis</th>
                    <th className="px-3 py-3 font-medium">Arms</th>
                    <th className="px-3 py-3 font-medium">Best Q</th>
                    <th className="px-3 py-3 font-medium">Val loss</th>
                    <th className="px-3 py-3 font-medium">Tokens</th>
                    <th className="px-3 py-3 font-medium">Config</th>
                  </tr>
                </thead>
                <tbody>
                  {[...saved].reverse().map((exp) => {
                    const best = [...exp.views].sort((a, b) => b.quality - a.quality)[0];
                    const on = compareIds.includes(exp.id);
                    return (
                      <tr key={exp.id} className="border-b border-border/80">
                        <td className="px-3 py-3">
                          <button
                            type="button"
                            onClick={() => toggleCompare(exp.id)}
                            className={`size-11 rounded-sm font-mono text-xs ${
                              on ? "bg-accent text-accent-fg" : "bg-surface-2 text-muted"
                            }`}
                            aria-pressed={on}
                          >
                            {on ? "On" : "Off"}
                          </button>
                        </td>
                        <td className="px-3 py-3 font-mono text-xs text-muted">
                          {new Date(exp.savedAt).toLocaleString()}
                        </td>
                        <td className="max-w-xs px-3 py-3 text-xs leading-relaxed">
                          {exp.hypothesis}
                        </td>
                        <td className="px-3 py-3">
                          <div className="flex flex-wrap gap-1">
                            {exp.views.map((v) => (
                              <Badge
                                key={v.id}
                                tone={
                                  v.mode === "adaptive"
                                    ? "good"
                                    : v.mode === "random"
                                      ? "warn"
                                      : "muted"
                                }
                              >
                                {v.mode}
                              </Badge>
                            ))}
                          </div>
                        </td>
                        <td className="px-3 py-3 font-mono tabular">
                          {best ? best.quality.toFixed(3) : "—"}
                        </td>
                        <td className="px-3 py-3 font-mono tabular">
                          {best ? formatLoss(best.valLoss) : "—"}
                        </td>
                        <td className="px-3 py-3 font-mono tabular text-muted">
                          {best ? formatCompact(best.tokens, 1) : "—"}
                        </td>
                        <td className="px-3 py-3 font-mono text-xs text-muted" title={exp.configHash}>
                          {exp.configHash.slice(0, 8)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {compared.length >= 2 ? (
              <div className="mt-6 grid gap-4 lg:grid-cols-2">
                <LossChart runs={compared} />
                <ParetoChart runs={compared} />
              </div>
            ) : (
              <p className="mt-4 text-xs text-muted">
                Select two or three rows to overlay their adaptive (or only) arms.
              </p>
            )}
            <button
              type="button"
              onClick={loadSaved}
              className="mt-4 text-xs text-muted underline-offset-2 hover:text-fg hover:underline"
            >
              Reload ledger from storage
            </button>
          </>
        )}
      </main>
    </AppShell>
  );
}

function label(exp: { savedAt: number; hypothesis: string }) {
  return new Date(exp.savedAt).toLocaleTimeString();
}
