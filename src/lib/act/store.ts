import { create } from "zustand";
import {
  type ControllerMode,
  type PresetId,
  type RunConfig,
  genomeHash,
} from "./genome";
import { SimEngine, type RunView } from "./engine";
import { cyrb53 } from "./rng";

export type LabPhase = "intro" | "lab";
export type Speed = 1 | 4 | 16;

export interface SavedExperiment {
  version: 1;
  id: string;
  savedAt: number;
  hypothesis: string;
  seed: number;
  preset: PresetId;
  configHash: string;
  architectureHashes: Record<string, string>;
  mutationCounts: Record<string, number>;
  views: RunView[];
}

interface LabState {
  phase: LabPhase;
  seed: number;
  preset: PresetId;
  allowModuleMutation: boolean;
  allowDepthMutation: boolean;
  playing: boolean;
  speed: Speed;
  fossilIndex: number | null;
  selectedId: string | null;
  runs: RunView[];
  saved: SavedExperiment[];
  compareIds: string[];
  startSingle: (mode: ControllerMode) => void;
  startExperiment0: () => void;
  togglePlay: () => void;
  setSpeed: (s: Speed) => void;
  select: (id: string) => void;
  setSeed: (seed: number) => void;
  setPreset: (p: PresetId) => void;
  setAllowModule: (v: boolean) => void;
  setAllowDepth: (v: boolean) => void;
  setFossilIndex: (i: number | null) => void;
  reset: () => void;
  saveCurrent: () => void;
  loadSaved: () => void;
  toggleCompare: (id: string) => void;
}

const engines = new Map<string, SimEngine>();
let raf = 0;
let acc = 0;
let last = 0;

function persistKey() {
  return "act-lab:experiments:v1";
}

function loadSaved(): SavedExperiment[] {
  if (typeof localStorage === "undefined") return [];
  try {
    const raw = localStorage.getItem(persistKey());
    if (!raw) return [];
    return JSON.parse(raw) as SavedExperiment[];
  } catch {
    return [];
  }
}

function writeSaved(items: SavedExperiment[]) {
  try {
    localStorage.setItem(persistKey(), JSON.stringify(items.slice(-24)));
  } catch {
    /* quota */
  }
}

function snapshot(): RunView[] {
  return [...engines.values()].map((e) => e.toView());
}

function stopLoop() {
  if (raf) cancelAnimationFrame(raf);
  raf = 0;
  last = 0;
}

function startLoop(get: () => LabState, set: (p: Partial<LabState>) => void) {
  stopLoop();
  const tick = (ts: number) => {
    const st = get();
    if (!st.playing) {
      raf = 0;
      return;
    }
    if (!last) last = ts;
    const dt = Math.min(48, ts - last);
    last = ts;
    acc += dt;
    const interval = 1000 / 60;
    let stepped = false;
    while (acc >= interval) {
      acc -= interval;
      const per = st.speed === 1 ? 2 : st.speed === 4 ? 8 : 28;
      let allDone = true;
      for (const eng of engines.values()) {
        if (!eng.done) {
          allDone = false;
          eng.status = "running";
          eng.stepMany(per);
        }
      }
      stepped = true;
      if (allDone) {
        const views = snapshot();
        set({ playing: false, runs: views, fossilIndex: null });
        stopLoop();
        return;
      }
    }
    if (stepped) {
      set({ runs: snapshot(), fossilIndex: null });
    }
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}

function makeConfig(s: LabState): Partial<RunConfig> {
  return {
    allowModuleMutation: s.allowModuleMutation,
    allowDepthMutation: s.allowDepthMutation,
  };
}

export const useLab = create<LabState>((set, get) => ({
  phase: "intro",
  seed: 42,
  preset: "supernet",
  allowModuleMutation: true,
  allowDepthMutation: false,
  playing: false,
  speed: 4,
  fossilIndex: null,
  selectedId: null,
  runs: [],
  saved: [],
  compareIds: [],

  startSingle: (mode) => {
    stopLoop();
    engines.clear();
    const s = get();
    const id = `${mode}-${s.seed}`;
    const eng = new SimEngine({
      id,
      name: modeLabel(mode),
      mode,
      seed: s.seed,
      preset: s.preset,
      config: makeConfig(s),
    });
    engines.set(id, eng);
    set({
      phase: "lab",
      playing: true,
      selectedId: id,
      fossilIndex: null,
      runs: snapshot(),
    });
    startLoop(get, set);
  },

  startExperiment0: () => {
    stopLoop();
    engines.clear();
    const s = get();
    const cfg = makeConfig(s);
    const specs: { id: string; mode: ControllerMode; name: string }[] = [
      { id: `frozen-${s.seed}`, mode: "frozen", name: "A · Fixed" },
      { id: `random-${s.seed}`, mode: "random", name: "B · Random churn" },
      { id: `adaptive-${s.seed}`, mode: "adaptive", name: "C · Adaptive" },
    ];
    for (const spec of specs) {
      engines.set(
        spec.id,
        new SimEngine({
          id: spec.id,
          name: spec.name,
          mode: spec.mode,
          seed: s.seed,
          preset: s.preset,
          config: cfg,
        }),
      );
    }
    set({
      phase: "lab",
      playing: true,
      selectedId: `adaptive-${s.seed}`,
      fossilIndex: null,
      runs: snapshot(),
    });
    startLoop(get, set);
  },

  togglePlay: () => {
    const s = get();
    if (s.phase !== "lab" || engines.size === 0) return;
    const allDone = [...engines.values()].every((e) => e.done);
    if (allDone) return;
    const playing = !s.playing;
    if (playing) {
      for (const e of engines.values()) {
        if (!e.done) e.status = "running";
      }
      set({ playing: true, runs: snapshot() });
      startLoop(get, set);
    } else {
      stopLoop();
      for (const e of engines.values()) {
        if (!e.done) e.status = "paused";
      }
      set({ playing: false, runs: snapshot() });
    }
  },

  setSpeed: (speed) => set({ speed }),
  select: (id) => set({ selectedId: id, fossilIndex: null }),
  setSeed: (seed) => set({ seed: Math.max(1, seed | 0) }),
  setPreset: (preset) => set({ preset }),
  setAllowModule: (allowModuleMutation) => set({ allowModuleMutation }),
  setAllowDepth: (allowDepthMutation) => set({ allowDepthMutation }),
  setFossilIndex: (fossilIndex) => set({ fossilIndex }),

  reset: () => {
    stopLoop();
    engines.clear();
    set({
      phase: "intro",
      playing: false,
      runs: [],
      selectedId: null,
      fossilIndex: null,
    });
  },

  saveCurrent: () => {
    const s = get();
    if (!s.runs.length) return;
    const item: SavedExperiment = {
      version: 1,
      id: `exp-${Date.now().toString(36)}`,
      savedAt: Date.now(),
      hypothesis:
        s.runs.length === 3
          ? "adaptive topology improves capability per FLOP vs random churn"
          : `single ${s.runs[0]?.mode} run`,
      seed: s.seed,
      preset: s.preset,
      configHash: cyrb53(JSON.stringify({
        seed: s.seed,
        preset: s.preset,
        allowModuleMutation: s.allowModuleMutation,
        allowDepthMutation: s.allowDepthMutation,
      })),
      architectureHashes: Object.fromEntries(
        s.runs.map((run) => [run.id, run.genomeHash]),
      ),
      mutationCounts: Object.fromEntries(
        s.runs.map((run) => [run.id, run.mutationCount]),
      ),
      views: s.runs,
    };
    const saved = [...s.saved, item].slice(-24);
    writeSaved(saved);
    set({ saved });
  },

  loadSaved: () => set({ saved: loadSaved() }),

  toggleCompare: (id) => {
    const cur = get().compareIds;
    const next = cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id].slice(-3);
    set({ compareIds: next });
  },
}));

export function modeLabel(mode: ControllerMode): string {
  if (mode === "frozen") return "Fixed";
  if (mode === "random") return "Random churn";
  return "Adaptive";
}

export function modeColor(mode: ControllerMode): string {
  if (mode === "frozen") return "var(--color-muted)";
  if (mode === "random") return "var(--color-warn)";
  return "var(--color-good)";
}

export function selectedRun(s: { runs: RunView[]; selectedId: string | null }) {
  return s.runs.find((r) => r.id === s.selectedId) ?? s.runs[0] ?? null;
}
