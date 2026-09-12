import type { RunView } from "@/lib/act/engine";
import { TASKS } from "@/lib/act/genome";
import { modeColor } from "@/lib/act/store";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  ScatterChart,
  Scatter,
  ZAxis,
} from "recharts";

export function LossChart({ runs }: { runs: RunView[] }) {
  const merged = mergeHistory(runs);
  return (
    <Panel title="Validation loss" hint="Lower is better · shock + adaptation visible as spikes">
      <div className="h-52 sm:h-56">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={merged} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--color-border)" vertical={false} />
            <XAxis
              dataKey="step"
              tick={{ fill: "var(--color-muted)", fontSize: 11, fontFamily: "IBM Plex Mono" }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              domain={["auto", "auto"]}
              width={40}
              tick={{ fill: "var(--color-muted)", fontSize: 11, fontFamily: "IBM Plex Mono" }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip content={<ChartTip />} />
            {runs.map((r) => (
              <Line
                key={r.id}
                type="monotone"
                dataKey={r.id}
                name={r.name}
                stroke={modeColor(r.mode)}
                strokeWidth={1.6}
                dot={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <SeriesLegend runs={runs} />
    </Panel>
  );
}

export function QualityChart({ runs }: { runs: RunView[] }) {
  const merged = mergeHistory(runs, "quality");
  return (
    <Panel title="Architecture quality" hint="Simulator score vs hidden attractor">
      <div className="h-52 sm:h-56">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={merged} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--color-border)" vertical={false} />
            <XAxis
              dataKey="step"
              tick={{ fill: "var(--color-muted)", fontSize: 11, fontFamily: "IBM Plex Mono" }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              domain={[0, 1]}
              width={36}
              tick={{ fill: "var(--color-muted)", fontSize: 11, fontFamily: "IBM Plex Mono" }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip content={<ChartTip digits={3} />} />
            {runs.map((r) => (
              <Line
                key={r.id}
                type="monotone"
                dataKey={r.id}
                name={r.name}
                stroke={modeColor(r.mode)}
                strokeWidth={1.6}
                dot={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  );
}

export function ProbeBars({ run }: { run: RunView }) {
  return (
    <Panel title="Capability probes" hint="Cheap diagnostics, not held-out benchmarks">
      <ul className="space-y-2.5">
        {TASKS.map((t) => {
          const v = run.probes[t] ?? 0;
          return (
            <li key={t}>
              <div className="mb-1 flex justify-between font-mono text-[0.6875rem] uppercase tracking-wider">
                <span className="text-muted">{t}</span>
                <span className="tabular text-fg">{(v * 100).toFixed(0)}</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-surface-2">
                <div
                  className="h-full rounded-full bg-accent transition-[width] duration-300 ease-out"
                  style={{ width: `${Math.max(2, v * 100)}%` }}
                />
              </div>
            </li>
          );
        })}
      </ul>
    </Panel>
  );
}

export function ParetoChart({ runs }: { runs: RunView[] }) {
  const data = runs.map((r) => ({
    id: r.id,
    name: r.name,
    x: r.flopsPerToken / 1e6,
    y: r.quality,
    z: 120,
    mode: r.mode,
  }));
  return (
    <Panel title="Quality vs FLOPs/token" hint="Pareto sketch — none should win on loss alone">
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--color-border)" />
            <XAxis
              dataKey="x"
              name="MFLOPs"
              tick={{ fill: "var(--color-muted)", fontSize: 11, fontFamily: "IBM Plex Mono" }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              dataKey="y"
              name="quality"
              domain={[0, 1]}
              width={36}
              tick={{ fill: "var(--color-muted)", fontSize: 11, fontFamily: "IBM Plex Mono" }}
              tickLine={false}
              axisLine={false}
            />
            <ZAxis dataKey="z" range={[80, 80]} />
            <Tooltip content={<ScatterTip />} />
            {data.map((d) => (
              <Scatter
                key={d.id}
                data={[d]}
                fill={modeColor(d.mode)}
                isAnimationActive={false}
                name={d.name}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  );
}

function Panel({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-lg bg-surface p-3 shadow-[var(--shadow-border)] sm:p-4">
      <h2 className="text-sm font-medium">{title}</h2>
      {hint ? <p className="mb-2 text-xs text-muted">{hint}</p> : null}
      {children}
    </section>
  );
}

function SeriesLegend({ runs }: { runs: RunView[] }) {
  if (runs.length < 2) return null;
  return (
    <ul className="mt-2 flex flex-wrap gap-3 text-xs text-muted">
      {runs.map((r) => (
        <li key={r.id} className="flex items-center gap-1.5">
          <span
            className="size-2 rounded-full"
            style={{ background: modeColor(r.mode) }}
          />
          {r.name}
        </li>
      ))}
    </ul>
  );
}

function mergeHistory(runs: RunView[], key: "valLoss" | "quality" = "valLoss") {
  const steps = new Map<number, Record<string, number>>();
  for (const r of runs) {
    for (const p of r.history) {
      const row = steps.get(p.step) ?? { step: p.step };
      row[r.id] = key === "valLoss" ? p.valLoss : p.quality;
      steps.set(p.step, row);
    }
  }
  return [...steps.values()].sort((a, b) => a.step - b.step);
}

function ChartTip({
  active,
  payload,
  label,
  digits = 3,
}: {
  active?: boolean;
  payload?: { color: string; name: string; value: number }[];
  label?: number;
  digits?: number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-sm bg-surface-2 px-2.5 py-2 text-xs shadow-[var(--shadow-border)]">
      <p className="mb-1 font-mono text-muted">step {label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {Number(p.value).toFixed(digits)}
        </p>
      ))}
    </div>
  );
}

function ScatterTip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: { name: string; x: number; y: number } }[];
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0]!.payload;
  return (
    <div className="rounded-sm bg-surface-2 px-2.5 py-2 text-xs shadow-[var(--shadow-border)]">
      <p className="font-medium">{d.name}</p>
      <p className="font-mono text-muted">
        {d.x.toFixed(1)} MFLOPs · Q {d.y.toFixed(3)}
      </p>
    </div>
  );
}
