import { createFileRoute } from "@tanstack/react-router";
import { AppShell } from "@/components/layout/shell";
import { StartScreen } from "@/components/lab/start-screen";
import { MetricsStrip } from "@/components/lab/metrics-strip";
import { TopologyStrip } from "@/components/lab/topology-strip";
import {
  LossChart,
  ParetoChart,
  ProbeBars,
  QualityChart,
} from "@/components/lab/charts";
import { Inspector } from "@/components/lab/inspector";
import { ControlBar } from "@/components/lab/control-bar";
import { Verdict } from "@/components/lab/verdict";
import { selectedRun, useLab } from "@/lib/act/store";
import { Toaster } from "sonner";
import { useEffect } from "react";

export const Route = createFileRoute("/")({ component: Home });

function Home() {
  const phase = useLab((s) => s.phase);
  const runs = useLab((s) => s.runs);
  const selectedId = useLab((s) => s.selectedId);
  const togglePlay = useLab((s) => s.togglePlay);
  const run = selectedRun({ runs, selectedId });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.code === "Space" && phase === "lab") {
        e.preventDefault();
        togglePlay();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase, togglePlay]);

  return (
    <AppShell>
      <Toaster
        theme="dark"
        position="bottom-right"
        toastOptions={{
          style: {
            background: "var(--color-surface)",
            color: "var(--color-fg)",
            border: "1px solid var(--color-border)",
          },
        }}
      />
      {phase === "intro" || !run ? (
        <StartScreen />
      ) : (
        <main className="space-y-4 px-4 py-5 sm:px-6">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="font-mono text-micro uppercase tracking-[0.18em] text-muted">
                {runs.length === 3 ? "Experiment 0" : run.name} · seed {run.seed} ·{" "}
                {run.preset}
              </p>
              <h1 className="text-xl font-medium tracking-tight sm:text-2xl">
                {run.status === "done" ? "Run complete" : "Training"}
              </h1>
            </div>
            <ControlBar />
          </div>
          <p className="max-w-2xl text-xs leading-relaxed text-muted">
            Laboratory-scale simulator of the V0 topology engine. Space pauses.
            Structural mutations occur on the slow clock, outside the gradient
            graph.
          </p>

          {runs.length === 3 ? <Verdict runs={runs} /> : null}
          <MetricsStrip run={run} />
          <TopologyStrip fossils={run.fossils} depth={run.genome.depth} />

          <div className="grid gap-4 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <LossChart runs={runs} />
            </div>
            <ProbeBars run={run} />
          </div>
          <div className="grid gap-4 lg:grid-cols-3">
            <QualityChart runs={runs} />
            <ParetoChart runs={runs} />
            <div className="rounded-lg bg-surface p-4 shadow-[var(--shadow-border)]">
              <h2 className="text-sm font-medium">Compute accounting</h2>
              <dl className="mt-3 space-y-2 font-mono text-xs tabular">
                <Row k="Training FLOPs" v={sci(run.trainingFlops)} />
                <Row k="Search FLOPs" v={sci(run.searchFlops)} />
                <Row k="CEC" v={run.cec.toFixed(3)} />
                <Row k="DAC" v={run.dac.toFixed(3)} />
                <Row k="Entropy H(G)" v={run.architectureEntropy.toFixed(3)} />
                <Row k="Exploration" v={run.exploration.toFixed(2)} />
                <Row k="Rollbacks" v={String(run.rollbackCount)} />
              </dl>
            </div>
          </div>
          <Inspector run={run} />
        </main>
      )}
    </AppShell>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="text-muted">{k}</dt>
      <dd>{v}</dd>
    </div>
  );
}

function sci(n: number): string {
  if (!n) return "0";
  return n.toExponential(2);
}
