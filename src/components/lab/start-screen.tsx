import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useLab } from "@/lib/act/store";
import type { ControllerMode, PresetId } from "@/lib/act/genome";
import { GitBranch, Pause, Play, Shuffle } from "lucide-react";

const PRESETS: { id: PresetId; label: string; hint: string }[] = [
  { id: "supernet", label: "Supernet", hint: "All attention + MoE — uninformed start" },
  { id: "dense", label: "Dense", hint: "Transformer baseline" },
  { id: "moe", label: "Static MoE", hint: "Attention + 8 experts" },
  { id: "qwen", label: "Qwen prior", hint: "3:1 GDN:attention, shared expert" },
];

const ARMS: {
  mode: ControllerMode;
  kicker: string;
  title: string;
  body: string;
  icon: typeof Pause;
}[] = [
  {
    mode: "frozen",
    kicker: "A",
    title: "Fixed baseline",
    body: "Weights train. Topology never moves. The human-designed control.",
    icon: Pause,
  },
  {
    mode: "random",
    kicker: "B",
    title: "Random churn",
    body: "Same mutation budget and clock, but components chosen at random.",
    icon: Shuffle,
  },
  {
    mode: "adaptive",
    kicker: "C",
    title: "Adaptive topology",
    body: "Importance-guided prune, grow, split, merge. The hypothesis.",
    icon: GitBranch,
  },
];

export function StartScreen() {
  const seed = useLab((s) => s.seed);
  const preset = useLab((s) => s.preset);
  const allowModule = useLab((s) => s.allowModuleMutation);
  const allowDepth = useLab((s) => s.allowDepthMutation);
  const setSeed = useLab((s) => s.setSeed);
  const setPreset = useLab((s) => s.setPreset);
  const setAllowModule = useLab((s) => s.setAllowModule);
  const setAllowDepth = useLab((s) => s.setAllowDepth);
  const startExperiment0 = useLab((s) => s.startExperiment0);
  const startSingle = useLab((s) => s.startSingle);

  return (
    <main className="px-4 py-10 sm:px-6 sm:py-16">
      <div className="stagger-in mx-auto max-w-3xl">
        <p className="font-mono text-[0.6875rem] uppercase tracking-[0.22em] text-muted">
          Research console · V0 · Laboratory scale
        </p>
        <h1 className="mt-4 text-4xl font-semibold tracking-tight text-fg sm:text-5xl">
          Adaptive Computational Topology
        </h1>
        <blockquote className="mt-6 max-w-2xl border-l border-accent/40 pl-4 text-base leading-relaxed text-muted sm:text-lg">
          A training process that jointly optimizes weights and computational
          topology can discover architectures with a better capability/FLOP
          frontier than manually designed architectures of equivalent capacity.
        </blockquote>
        <p className="mt-5 max-w-2xl text-sm leading-relaxed text-muted">
          This console implements the V0 mechanism: a serializable architecture
          genome, discrete topology operators, an importance system, and a
          rule-based controller. Experiment 0 asks the only question that
          matters first — does information-guided evolution beat the same amount
          of random topology churn?
        </p>
      </div>

      <div className="mx-auto mt-10 grid max-w-5xl gap-3 sm:grid-cols-3">
        {ARMS.map((arm) => {
          const Icon = arm.icon;
          return (
            <button
              key={arm.mode}
              type="button"
              onClick={() => startSingle(arm.mode)}
              className="rounded-xl bg-surface p-5 text-left shadow-[var(--shadow-border)] transition-[box-shadow,transform] duration-150 hover:shadow-[var(--shadow-border-hover)] active:scale-[0.99]"
            >
              <div className="flex items-center justify-between">
                <Badge tone={arm.mode === "adaptive" ? "good" : arm.mode === "random" ? "warn" : "muted"}>
                  {arm.kicker}
                </Badge>
                <Icon className="size-4 text-muted" />
              </div>
              <h2 className="mt-3 text-base font-medium text-fg">{arm.title}</h2>
              <p className="mt-1.5 text-sm leading-relaxed text-muted">{arm.body}</p>
              <p className="mt-4 font-mono text-[0.6875rem] uppercase tracking-wider text-subtle">
                Run {arm.title.toLowerCase()}
              </p>
            </button>
          );
        })}
      </div>

      <div className="mx-auto mt-8 max-w-5xl rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="flex-1">
            <p className="font-mono text-[0.6875rem] uppercase tracking-[0.18em] text-muted">
              Experiment 0
            </p>
            <h2 className="mt-1 text-xl font-medium">Three arms, one budget</h2>
            <p className="mt-2 max-w-xl text-sm leading-relaxed text-muted">
              Identical tokens, mutation count, compute ceiling, and seed. Success
              is C beats B. If they tie, the importance subsystem is not earning
              its complexity.
            </p>
          </div>
          <Button size="lg" onClick={startExperiment0} className="w-full sm:w-auto">
            <Play className="size-4" />
            Run Experiment 0
          </Button>
        </div>

        <div className="mt-6 grid gap-4 border-t border-border pt-5 sm:grid-cols-2 lg:grid-cols-4">
          <label className="block">
            <span className="font-mono text-[0.6875rem] uppercase tracking-wider text-muted">
              Seed
            </span>
            <input
              type="number"
              min={1}
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value) || 1)}
              className="mt-1.5 h-11 w-full rounded-sm bg-surface-2 px-3 font-mono text-sm text-fg shadow-[var(--shadow-border)] outline-none focus:ring-2 focus:ring-ring/60"
            />
          </label>
          <fieldset className="sm:col-span-1 lg:col-span-3">
            <legend className="font-mono text-[0.6875rem] uppercase tracking-wider text-muted">
              Initial genome
            </legend>
            <div className="mt-1.5 flex flex-wrap gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => setPreset(p.id)}
                  className={`h-11 rounded-sm px-3 text-sm transition-colors duration-150 ${
                    preset === p.id
                      ? "bg-accent text-accent-fg"
                      : "bg-surface-2 text-muted hover:text-fg"
                  }`}
                  title={p.hint}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </fieldset>
        </div>

        <div className="mt-4 flex flex-wrap gap-3">
          <Toggle
            label="Module-type evolution"
            hint="Exp 4 — attention / GDN / hybrid"
            on={allowModule}
            onChange={setAllowModule}
          />
          <Toggle
            label="Depth prune / grow"
            hint="Exp 2 — layer count"
            on={allowDepth}
            onChange={setAllowDepth}
          />
        </div>
      </div>
    </main>
  );
}

function Toggle({
  label,
  hint,
  on,
  onChange,
}: {
  label: string;
  hint: string;
  on: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!on)}
      className="flex min-h-11 items-center gap-3 rounded-md px-3 py-2 text-left hover:bg-surface-2"
    >
      <span
        className={`relative h-5 w-9 rounded-full transition-colors duration-150 ${
          on ? "bg-accent" : "bg-surface-2 shadow-[var(--shadow-border)]"
        }`}
      >
        <span
          className={`absolute top-0.5 size-4 rounded-full bg-fg transition-transform duration-150 ${
            on ? "translate-x-4 bg-accent-fg" : "translate-x-0.5"
          }`}
        />
      </span>
      <span>
        <span className="block text-sm text-fg">{label}</span>
        <span className="block text-xs text-muted">{hint}</span>
      </span>
    </button>
  );
}
