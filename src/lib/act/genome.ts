import { cyrb53, Rng } from "./rng";

export const TASKS = [
  "language",
  "arithmetic",
  "logic",
  "code",
  "knowledge",
  "longrange",
] as const;
export type Task = (typeof TASKS)[number];

export type ModuleType = "attention" | "gdn" | "hybrid";
export type FfnType = "dense" | "moe";
export type ControllerMode = "frozen" | "random" | "adaptive";
export type RunStatus = "idle" | "running" | "paused" | "done";
export type MutationOp =
  | "prune"
  | "grow"
  | "split"
  | "merge"
  | "rewire"
  | "module"
  | "rollback";

export type Lifecycle =
  | "new"
  | "emerging"
  | "active"
  | "stable"
  | "essential"
  | "uncertain"
  | "redundant"
  | "dying";

export const LIFECYCLE_ORDER: Lifecycle[] = [
  "new",
  "emerging",
  "active",
  "stable",
  "essential",
  "uncertain",
  "redundant",
  "dying",
];

export interface ExpertGene {
  id: string;
  parentId: string | null;
  birthStep: number;
  specialty: number[];
  width: number;
}

export interface LayerGene {
  id: string;
  index: number;
  module: ModuleType;
  heads: number;
  kvHeads: number;
  ffn: FfnType;
  ffnWidth: number;
  experts: ExpertGene[];
  activeExperts: number;
  sharedExpert: boolean;
  residualPaths: number;
  enabled: boolean;
}

export interface GenomeGlobal {
  maxParams: number;
  maxActiveParams: number;
  maxFlopsPerToken: number;
  earlyExit: boolean;
}

export interface ArchitectureGenome {
  version: 1;
  seed: number;
  depth: number;
  hiddenSize: number;
  vocabSize: number;
  context: number;
  layers: LayerGene[];
  global: GenomeGlobal;
}

export interface CompactLayer {
  module: ModuleType;
  ffn: FfnType;
  experts: number;
  active: number;
  enabled: boolean;
  heads: number;
}

export interface FossilFrame {
  step: number;
  tokens: number;
  loss: number;
  quality: number;
  params: number;
  activeParams: number;
  layers: CompactLayer[];
}

export interface MetricPoint {
  step: number;
  tokens: number;
  loss: number;
  valLoss: number;
  quality: number;
  params: number;
  activeParams: number;
  flopsPerToken: number;
  cec: number;
}

export interface MutationEvent {
  step: number;
  tokens: number;
  op: MutationOp;
  target: string;
  reason: string;
  accepted: boolean;
  lossBefore: number;
  lossAfter?: number;
}

export interface ImportanceView {
  id: string;
  kind: "layer" | "expert" | "head";
  label: string;
  activation: number;
  routing: number;
  gradient: number;
  fisher: number;
  uniqueness: number;
  stability: number;
  lifecycle: Lifecycle;
  age: number;
}

export interface ExpertView {
  id: string;
  layerId: string;
  layerIndex: number;
  specialty: number[];
  routing: number;
  uniqueness: number;
  lifecycle: Lifecycle;
  parentId: string | null;
  age: number;
}

export interface RunConfig {
  seed: number;
  mode: ControllerMode;
  name: string;
  maxSteps: number;
  tokensPerStep: number;
  mutationInterval: number;
  statsInterval: number;
  allowModuleMutation: boolean;
  allowDepthMutation: boolean;
  batchTokens: number;
}

export const DEFAULT_CONFIG: Omit<RunConfig, "seed" | "mode" | "name"> = {
  maxSteps: 2400,
  tokensPerStep: 65536,
  mutationInterval: 80,
  statsInterval: 5,
  allowModuleMutation: false,
  allowDepthMutation: false,
  batchTokens: 65536,
};

export function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

export function l2normalize(v: number[]): number[] {
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map((x) => x / n);
}

export function cosine(a: number[], b: number[]): number {
  let d = 0;
  let na = 0;
  let nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    d += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return d / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

function randomSpecialty(rng: Rng): number[] {
  return l2normalize(TASKS.map(() => Math.abs(rng.gaussian(0.6, 0.45))));
}

let seq = 1;
export function uid(prefix: string, rng: Rng): string {
  seq += 1;
  return `${prefix}-${(rng.int(1e9) + seq).toString(36)}`;
}

export function createLayer(
  rng: Rng,
  index: number,
  opts: {
    module: ModuleType;
    ffn: FfnType;
    hiddenSize: number;
    heads?: number;
    kvHeads?: number;
    ffnWidth?: number;
    expertCount?: number;
    activeExperts?: number;
    sharedExpert?: boolean;
    expertWidth?: number;
  },
): LayerGene {
  const heads = opts.heads ?? 16;
  const experts: ExpertGene[] = [];
  if (opts.ffn === "moe") {
    const count = opts.expertCount ?? 8;
    const width = opts.expertWidth ?? opts.ffnWidth ?? opts.hiddenSize;
    for (let i = 0; i < count; i++) {
      experts.push({
        id: uid("e", rng),
        parentId: null,
        birthStep: 0,
        specialty: randomSpecialty(rng),
        width,
      });
    }
  }
  return {
    id: uid("L", rng),
    index,
    module: opts.module,
    heads,
    kvHeads: opts.kvHeads ?? Math.max(1, Math.floor(heads / 4)),
    ffn: opts.ffn,
    ffnWidth: opts.ffnWidth ?? opts.hiddenSize * 4,
    experts,
    activeExperts: opts.ffn === "moe" ? (opts.activeExperts ?? 2) : 0,
    sharedExpert: opts.sharedExpert ?? false,
    residualPaths: 1,
    enabled: true,
  };
}

export type PresetId =
  | "dense"
  | "moe"
  | "qwen"
  | "supernet"
  | "hybrid_static";

export function createGenome(
  seed: number,
  preset: PresetId = "supernet",
  depth = 12,
  hiddenSize = 1024,
): ArchitectureGenome {
  const rng = new Rng(seed);
  const layers: LayerGene[] = [];
  for (let i = 0; i < depth; i++) {
    const everyFourth = i % 4 === 3;
    let module: ModuleType = "attention";
    let ffn: FfnType = "dense";
    let expertCount = 0;
    let shared = false;
    let ffnWidth = hiddenSize * 4;

    if (preset === "dense") {
      module = "attention";
      ffn = "dense";
    } else if (preset === "moe") {
      module = "attention";
      ffn = "moe";
      expertCount = 8;
      ffnWidth = hiddenSize;
    } else if (preset === "qwen") {
      module = everyFourth ? "attention" : "gdn";
      ffn = "moe";
      expertCount = 8;
      shared = true;
      ffnWidth = hiddenSize;
    } else if (preset === "hybrid_static") {
      module = everyFourth ? "attention" : i % 2 === 0 ? "gdn" : "hybrid";
      ffn = i % 3 === 0 ? "dense" : "moe";
      expertCount = ffn === "moe" ? 8 : 0;
      ffnWidth = ffn === "moe" ? hiddenSize : hiddenSize * 4;
    } else {
      // supernet — all attention + oversized MoE; search starts uninformed
      module = "attention";
      ffn = "moe";
      expertCount = 12;
      ffnWidth = hiddenSize;
    }

    layers.push(
      createLayer(rng, i, {
        module,
        ffn,
        hiddenSize,
        ffnWidth,
        expertCount,
        activeExperts: ffn === "moe" ? 2 : 0,
        sharedExpert: shared,
        expertWidth: hiddenSize,
      }),
    );
  }

  const genome: ArchitectureGenome = {
    version: 1,
    seed,
    depth,
    hiddenSize,
    vocabSize: 32768,
    context: 1024,
    layers,
    global: {
      maxParams: 350_000_000,
      maxActiveParams: 160_000_000,
      maxFlopsPerToken: 2.4e9,
      earlyExit: false,
    },
  };
  const stats = parameterStats(genome);
  genome.global.maxParams = Math.round(Math.max(350_000_000, stats.params * 1.12));
  genome.global.maxActiveParams = Math.round(
    Math.max(160_000_000, stats.activeParams * 1.25),
  );
  return genome;
}

export function cloneGenome(g: ArchitectureGenome): ArchitectureGenome {
  return structuredClone(g);
}

export interface GenomeValidation {
  valid: boolean;
  errors: string[];
}

export function validateGenome(g: ArchitectureGenome): GenomeValidation {
  const errors: string[] = [];
  const enabled = g.layers.filter((layer) => layer.enabled).length;

  if (g.version !== 1) errors.push(`unsupported genome version ${g.version}`);
  if (g.depth !== g.layers.length) errors.push("depth does not match layer count");
  if (enabled < 1) errors.push("genome has no enabled layers");
  if (!Number.isFinite(g.hiddenSize) || g.hiddenSize <= 0) {
    errors.push("hidden size is invalid");
  }

  const layerIds = new Set<string>();
  for (const [index, layer] of g.layers.entries()) {
    if (layer.index !== index) errors.push(`layer ${layer.id} has stale index`);
    if (layerIds.has(layer.id)) errors.push(`duplicate layer id ${layer.id}`);
    layerIds.add(layer.id);
    if (layer.heads < 1 || layer.kvHeads < 1 || layer.kvHeads > layer.heads) {
      errors.push(`invalid head shape for ${layer.id}`);
    }
    if (layer.ffn === "dense" && layer.experts.length !== 0) {
      errors.push(`dense layer ${layer.id} has experts`);
    }
    if (layer.ffn === "moe") {
      if (layer.experts.length < 2) errors.push(`MoE layer ${layer.id} has too few experts`);
      if (layer.activeExperts < 1 || layer.activeExperts > layer.experts.length) {
        errors.push(`invalid active expert count for ${layer.id}`);
      }
    } else if (layer.activeExperts !== 0) {
      errors.push(`dense layer ${layer.id} has active experts`);
    }
    const expertIds = new Set<string>();
    for (const expert of layer.experts) {
      if (expertIds.has(expert.id)) errors.push(`duplicate expert id ${expert.id}`);
      expertIds.add(expert.id);
      if (!Number.isFinite(expert.width) || expert.width <= 0) {
        errors.push(`invalid width for ${expert.id}`);
      }
      if (expert.specialty.length !== TASKS.length || expert.specialty.some((x) => !Number.isFinite(x))) {
        errors.push(`invalid specialty for ${expert.id}`);
      }
    }
  }

  const stats = parameterStats(g);
  if (!Number.isFinite(stats.params) || !Number.isFinite(stats.activeParams)) {
    errors.push("parameter statistics are not finite");
  }
  if (stats.params > g.global.maxParams || stats.activeParams > g.global.maxActiveParams) {
    errors.push("parameter budget exceeded");
  }
  return { valid: errors.length === 0, errors };
}

export function compactLayers(g: ArchitectureGenome): CompactLayer[] {
  return g.layers.map((l) => ({
    module: l.module,
    ffn: l.ffn,
    experts: l.experts.length,
    active: l.activeExperts,
    enabled: l.enabled,
    heads: l.heads,
  }));
}

export function serializeGenome(g: ArchitectureGenome): string {
  return JSON.stringify(g, null, 2);
}

export function genomeHash(g: ArchitectureGenome): string {
  const canon = {
    version: g.version,
    depth: g.depth,
    hiddenSize: g.hiddenSize,
    vocabSize: g.vocabSize,
    context: g.context,
    layers: g.layers.map((l) => ({
      module: l.module,
      heads: l.heads,
      kvHeads: l.kvHeads,
      ffn: l.ffn,
      ffnWidth: l.ffnWidth,
      experts: l.experts.length,
      activeExperts: l.activeExperts,
      sharedExpert: l.sharedExpert,
      residualPaths: l.residualPaths,
      enabled: l.enabled,
    })),
  };
  return cyrb53(JSON.stringify(canon));
}

export function parameterStats(g: ArchitectureGenome): {
  params: number;
  activeParams: number;
  embed: number;
} {
  const h = g.hiddenSize;
  const embed = g.vocabSize * h; // tied embeddings
  let params = embed;
  let active = embed;
  for (const layer of g.layers) {
    if (!layer.enabled) continue;
    const s = layerParams(layer, h);
    params += s.params;
    active += s.active;
  }
  return { params, activeParams: active, embed };
}

export function layerParams(
  layer: LayerGene,
  hidden: number,
): { params: number; active: number } {
  const h = hidden;
  let params = 4 * h; // pre/post norms
  if (layer.module === "attention" || layer.module === "hybrid") {
    const headDim = h / Math.max(layer.heads, 1);
    params += h * h; // Q
    params += layer.kvHeads * headDim * h * 2; // K, V
    params += h * h; // O
  }
  if (layer.module === "gdn" || layer.module === "hybrid") {
    params += 3 * h * h + 4 * h;
  }
  if (layer.ffn === "dense") {
    params += 3 * h * layer.ffnWidth;
  } else {
    params += h * Math.max(layer.experts.length, 1); // router
    const width =
      layer.experts[0]?.width ?? layer.ffnWidth ?? hidden;
    params += layer.experts.length * 3 * h * width;
    if (layer.sharedExpert) params += 3 * h * width;
  }
  let active = params;
  if (layer.ffn === "moe" && layer.experts.length > 0) {
    const width = layer.experts[0]!.width;
    const expertFull = layer.experts.length * 3 * h * width;
    const expertActive =
      Math.min(layer.activeExperts, layer.experts.length) * 3 * h * width +
      (layer.sharedExpert ? 3 * h * width : 0);
    active = params - expertFull + expertActive;
  }
  return { params, active };
}

/** FLOPs per token, approximate forward pass. */
export function flopsPerToken(g: ArchitectureGenome): number {
  const h = g.hiddenSize;
  const seq = g.context;
  let flops = 2 * g.vocabSize * h; // embed + lm head
  for (const layer of g.layers) {
    if (!layer.enabled) continue;
    if (layer.module === "attention" || layer.module === "hybrid") {
      const headDim = h / Math.max(layer.heads, 1);
      const kv = layer.kvHeads * headDim;
      flops += 2 * h * h + 2 * h * kv * 2; // Q, K, V, O
      flops += 2 * seq * h; // attn scores amortized per token
    }
    if (layer.module === "gdn" || layer.module === "hybrid") {
      flops += 6 * h * h;
    }
    if (layer.ffn === "dense") {
      flops += 6 * h * layer.ffnWidth;
    } else if (layer.experts.length) {
      const w = layer.experts[0]!.width;
      const k = Math.min(layer.activeExperts, layer.experts.length);
      flops += 2 * h * layer.experts.length; // router
      flops += 6 * h * w * k;
      if (layer.sharedExpert) flops += 6 * h * w;
    }
  }
  return flops;
}

export function reindex(genome: ArchitectureGenome): void {
  genome.layers.forEach((l, i) => {
    l.index = i;
  });
  genome.depth = genome.layers.length;
}

export const MODULE_AFFINITY: Record<ModuleType, number[]> = {
  attention: [0.52, 0.48, 0.92, 0.66, 0.74, 1.0],
  gdn: [0.96, 0.55, 0.42, 0.5, 0.58, 0.28],
  hybrid: [0.74, 0.52, 0.68, 0.58, 0.66, 0.64],
};

/** Hidden attractor used by the laboratory simulator — 3:1 GDN:attention, MoE on GDN. */
export function wantsAttention(index: number, depth: number): boolean {
  if (index === 0 || index === depth - 1) return true;
  return index % 4 === 3;
}
