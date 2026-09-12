import type { CompactLayer, FossilFrame, ModuleType } from "@/lib/act/genome";
import { cn } from "@/lib/utils";
import { useLab } from "@/lib/act/store";

const MODULE_CLASS: Record<ModuleType, string> = {
  attention: "bg-att",
  gdn: "bg-gdn",
  hybrid: "bg-hybrid",
};

export function TopologyStrip({
  fossils,
  depth,
}: {
  fossils: FossilFrame[];
  depth: number;
}) {
  const fossilIndex = useLab((s) => s.fossilIndex);
  const setFossilIndex = useLab((s) => s.setFossilIndex);
  const frames = fossils.length ? fossils : [];
  const active = fossilIndex ?? frames.length - 1;
  const rows = Math.max(depth, frames[0]?.layers.length ?? 12);

  if (!frames.length) {
    return (
      <div className="flex h-48 items-center justify-center rounded-lg bg-surface text-sm text-muted shadow-[var(--shadow-border)]">
        No fossils yet.
      </div>
    );
  }

  return (
    <div className="rounded-lg bg-surface p-3 shadow-[var(--shadow-border)] sm:p-4">
      <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
        <div>
          <h2 className="text-sm font-medium">Architecture fossils</h2>
          <p className="text-xs text-muted">
            Step {frames[active]?.step.toLocaleString()} · layer type over
            structural time
          </p>
        </div>
        <Legend />
      </div>
      <div className="-mx-1 overflow-x-auto pb-2">
        <div
          className="grid min-w-max gap-0.5"
          style={{
            gridTemplateColumns: `2.5rem repeat(${frames.length}, minmax(1.35rem, 1.75rem))`,
          }}
        >
          {Array.from({ length: rows }, (_, li) => (
            <Row
              key={li}
              li={li}
              frames={frames}
              active={active}
              onPick={setFossilIndex}
            />
          ))}
        </div>
      </div>
      <input
        type="range"
        min={0}
        max={frames.length - 1}
        value={active}
        onChange={(e) => setFossilIndex(Number(e.target.value))}
        className="mt-3 h-11 w-full accent-accent"
        aria-label="Scrub architecture fossils"
      />
    </div>
  );
}

function Row({
  li,
  frames,
  active,
  onPick,
}: {
  li: number;
  frames: FossilFrame[];
  active: number;
  onPick: (i: number) => void;
}) {
  return (
    <>
      <div className="flex h-6 items-center font-mono text-[0.625rem] text-subtle">
        L{String(li).padStart(2, "0")}
      </div>
      {frames.map((f, fi) => {
        const layer: CompactLayer | undefined = f.layers[li];
        const isActive = fi === active;
        return (
          <button
            key={`${fi}-${li}`}
            type="button"
            onClick={() => onPick(fi)}
            title={
              layer
                ? `${layer.module} · ${layer.ffn}${layer.ffn === "moe" ? ` ${layer.experts}e/${layer.active}k` : ""}`
                : "—"
            }
            className={cn(
              "relative h-6 rounded-[3px] transition-opacity duration-150",
              layer && layer.enabled
                ? MODULE_CLASS[layer.module]
                : "bg-surface-2",
              !layer?.enabled && "opacity-30",
              isActive && "ring-1 ring-fg/80 ring-offset-1 ring-offset-surface",
            )}
            style={
              layer?.enabled
                ? { opacity: 0.55 + 0.45 * Math.min(1, layer.experts / 12) }
                : undefined
            }
          >
            {layer?.ffn === "moe" && layer.enabled ? (
              <span className="absolute inset-0 m-auto size-1.5 rounded-full bg-bg/80" />
            ) : null}
          </button>
        );
      })}
    </>
  );
}

function Legend() {
  const items: { cls: string; label: string }[] = [
    { cls: "bg-att", label: "Attention" },
    { cls: "bg-gdn", label: "GDN" },
    { cls: "bg-hybrid", label: "Hybrid" },
  ];
  return (
    <ul className="flex flex-wrap items-center gap-3 text-[0.6875rem] text-muted">
      {items.map((it) => (
        <li key={it.label} className="flex items-center gap-1.5">
          <span className={cn("size-2.5 rounded-[2px]", it.cls)} />
          {it.label}
        </li>
      ))}
      <li className="flex items-center gap-1.5">
        <span className="relative size-2.5 rounded-[2px] bg-att">
          <span className="absolute inset-0 m-auto size-1 rounded-full bg-bg" />
        </span>
        MoE
      </li>
    </ul>
  );
}
