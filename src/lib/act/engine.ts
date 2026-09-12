import {
  type ArchitectureGenome,
  type CompactLayer,
  type ControllerMode,
  type ExpertView,
  type FossilFrame,
  type ImportanceView,
  type LayerGene,
  type Lifecycle,
  type MetricPoint,
  type MutationEvent,
  type PresetId,
  type RunConfig,
  type RunStatus,
  type Task,
  TASKS,
  MODULE_AFFINITY,
  DEFAULT_CONFIG,
  clamp,
  cloneGenome,
  compactLayers,
  cosine,
  createGenome,
  flopsPerToken,
  genomeHash,
  l2normalize,
  parameterStats,
  validateGenome,
  wantsAttention,
} from "./genome";
import {
  growExpert,
  growLayer,
  mergeExperts,
  mutateModule,
  pruneExpert,
  pruneLayer,
  rewireLayer,
  splitExpert,
} from "./mutations";
import { Rng } from "./rng";

interface ImportanceRec {
  id: string;
  kind: "layer" | "expert" | "head";
  activation: number;
  routing: number;
  gradient: number;
  fisher: number;
  uniqueness: number;
  stability: number;
  lifecycle: Lifecycle;
  age: number;
  history: number[];
}

export interface EngineCheckpoint {
  version: 1;
  opts: {
    id: string;
    name: string;
    mode: ControllerMode;
    seed: number;
    preset: PresetId;
    config: RunConfig;
  };
  genome: ArchitectureGenome;
  stepCount: number;
  tokens: number;
  loss: number;
  valLoss: number;
  quality: number;
  trainingFlops: number;
  searchFlops: number;
  mutationCount: number;
  rollbackCount: number;
  acceptCount: number;
  exploration: number;
  status: RunStatus;
  fossils: FossilFrame[];
  mutations: MutationEvent[];
  history: MetricPoint[];
  probes: Record<Task, number>;
  importance: ImportanceRec[];
  routingEma: Array<[string, number]>;
  rng: { controller: number; route: number; data: number };
  shock: number;
  pending: SimEngine["pending"];
}

export interface RunView {
  id: string;
  name: string;
  mode: ControllerMode;
  seed: number;
  preset: PresetId;
  step: number;
  maxSteps: number;
  tokens: number;
  loss: number;
  valLoss: number;
  quality: number;
  cec: number;
  dac: number;
  params: number;
  activeParams: number;
  flopsPerToken: number;
  trainingFlops: number;
  searchFlops: number;
  mutationCount: number;
  rollbackCount: number;
  genome: ArchitectureGenome;
  genomeHash: string;
  fossils: FossilFrame[];
  mutations: MutationEvent[];
  history: MetricPoint[];
  importance: ImportanceView[];
  experts: ExpertView[];
  probes: Record<Task, number>;
  status: RunStatus;
  lastMutation: MutationEvent | null;
  exploration: number;
  architectureEntropy: number;
}

const TASK_WEIGHTS: Record<Task, number> = {
  language: 0.28,
  arithmetic: 0.12,
  logic: 0.14,
  code: 0.16,
  knowledge: 0.14,
  longrange: 0.16,
};

function softmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  const ex = logits.map((x) => Math.exp(x - m));
  const s = ex.reduce((a, b) => a + b, 0) || 1;
  return ex.map((x) => x / s);
}

function entropy(p: number[]): number {
  let h = 0;
  for (const x of p) {
    if (x > 1e-12) h -= x * Math.log(x);
  }
  return h;
}

function specialization(utils: number[]): number {
  if (utils.length < 2) return 0;
  const s = utils.reduce((a, b) => a + b, 0) || 1;
  const p = utils.map((u) => u / s);
  const h = entropy(p);
  const maxH = Math.log(p.length);
  const even = 1 - h / (maxH || 1);
  const collapsed = Math.max(...p) > 0.85 ? 0.35 : 1;
  return clamp(even * collapsed, 0, 1);
}

export class SimEngine {
  readonly id: string;
  readonly name: string;
  readonly mode: ControllerMode;
  readonly preset: PresetId;
  readonly config: RunConfig;
  genome: ArchitectureGenome;
  private ctrlRng: Rng;
  private routeRng: Rng;
  private dataRng: Rng;
  stepCount = 0;
  tokens = 0;
  loss = 4.55;
  valLoss = 4.62;
  quality = 0.22;
  trainingFlops = 0;
  searchFlops = 0;
  mutationCount = 0;
  rollbackCount = 0;
  acceptCount = 0;
  exploration = 1;
  status: RunStatus = "idle";
  fossils: FossilFrame[] = [];
  mutations: MutationEvent[] = [];
  history: MetricPoint[] = [];
  probes: Record<Task, number> = {
    language: 0.12,
    arithmetic: 0.08,
    logic: 0.07,
    code: 0.09,
    knowledge: 0.1,
    longrange: 0.06,
  };
  private importance = new Map<string, ImportanceRec>();
  private routingEma = new Map<string, number>();
  private shock = 0;
  private pending: {
    snapshot: ArchitectureGenome;
    event: MutationEvent;
    until: number;
    preLoss: number;
  } | null = null;

  constructor(opts: {
    id: string;
    name: string;
    mode: ControllerMode;
    seed: number;
    preset?: PresetId;
    config?: Partial<RunConfig>;
  }) {
    this.id = opts.id;
    this.name = opts.name;
    this.mode = opts.mode;
    this.preset = opts.preset ?? "supernet";
    this.config = {
      ...DEFAULT_CONFIG,
      seed: opts.seed,
      mode: opts.mode,
      name: opts.name,
      ...opts.config,
    };
    this.ctrlRng = new Rng(opts.seed ^ (this.mode === "random" ? 0x51ed : 0xac70));
    this.routeRng = new Rng(opts.seed ^ 0x51ed);
    this.dataRng = new Rng(opts.seed);
    this.genome = createGenome(opts.seed, this.preset);
    this.ensureImportance();
    this.recordHistory();
    this.recordFossil();
    this.quality = this.scoreArchitecture();
    this.loss = this.computeLoss();
    this.valLoss = this.loss + 0.05;
  }

  toCheckpoint(): EngineCheckpoint {
    return structuredClone({
      version: 1 as const,
      opts: {
        id: this.id,
        name: this.name,
        mode: this.mode,
        seed: this.config.seed,
        preset: this.preset,
        config: this.config,
      },
      genome: this.genome,
      stepCount: this.stepCount,
      tokens: this.tokens,
      loss: this.loss,
      valLoss: this.valLoss,
      quality: this.quality,
      trainingFlops: this.trainingFlops,
      searchFlops: this.searchFlops,
      mutationCount: this.mutationCount,
      rollbackCount: this.rollbackCount,
      acceptCount: this.acceptCount,
      exploration: this.exploration,
      status: this.status,
      fossils: this.fossils,
      mutations: this.mutations,
      history: this.history,
      probes: this.probes,
      importance: [...this.importance.values()],
      routingEma: [...this.routingEma.entries()],
      rng: {
        controller: this.ctrlRng.state,
        route: this.routeRng.state,
        data: this.dataRng.state,
      },
      shock: this.shock,
      pending: this.pending,
    });
  }

  static fromCheckpoint(checkpoint: EngineCheckpoint): SimEngine {
    if (checkpoint.version !== 1) {
      throw new Error(`unsupported engine checkpoint version ${checkpoint.version}`);
    }
    const engine = new SimEngine({ ...checkpoint.opts });
    engine.genome = structuredClone(checkpoint.genome);
    engine.stepCount = checkpoint.stepCount;
    engine.tokens = checkpoint.tokens;
    engine.loss = checkpoint.loss;
    engine.valLoss = checkpoint.valLoss;
    engine.quality = checkpoint.quality;
    engine.trainingFlops = checkpoint.trainingFlops;
    engine.searchFlops = checkpoint.searchFlops;
    engine.mutationCount = checkpoint.mutationCount;
    engine.rollbackCount = checkpoint.rollbackCount;
    engine.acceptCount = checkpoint.acceptCount;
    engine.exploration = checkpoint.exploration;
    engine.status = checkpoint.status;
    engine.fossils = structuredClone(checkpoint.fossils);
    engine.mutations = structuredClone(checkpoint.mutations);
    engine.history = structuredClone(checkpoint.history);
    engine.probes = structuredClone(checkpoint.probes);
    engine.importance = new Map(
      checkpoint.importance.map((record) => [record.id, structuredClone(record)]),
    );
    engine.routingEma = new Map(checkpoint.routingEma);
    engine.ctrlRng.state = checkpoint.rng.controller;
    engine.routeRng.state = checkpoint.rng.route;
    engine.dataRng.state = checkpoint.rng.data;
    engine.shock = checkpoint.shock;
    engine.pending = structuredClone(checkpoint.pending);
    return engine;
  }

  get done(): boolean {
    return this.stepCount >= this.config.maxSteps;
  }

  stepMany(n: number): void {
    const cap = Math.min(n, this.config.maxSteps - this.stepCount);
    for (let i = 0; i < cap; i++) this.step();
  }

  step(): void {
    if (this.done) {
      this.status = "done";
      return;
    }
    this.stepCount += 1;
    this.tokens += this.config.tokensPerStep;
    this.exploration = clamp(1 - this.stepCount / this.config.maxSteps, 0.08, 1);

    const task = this.sampleTaskMix();
    this.updateRouting(task);
    this.hebbian(task);

    const flops = flopsPerToken(this.genome);
    this.trainingFlops += flops * this.config.tokensPerStep;

    this.quality = this.scoreArchitecture();
    this.shock *= 0.965;
    this.loss = this.computeLoss();
    this.valLoss =
      this.loss + 0.035 + 0.03 * (1 - this.quality) + this.dataRng.gaussian(0, 0.004);

    this.updateProbes();

    if (this.stepCount % this.config.statsInterval === 0) {
      this.updateImportance();
    }

    if (this.pending && this.stepCount >= this.pending.until) {
      this.resolvePending();
    }

    const due =
      this.mode !== "frozen" &&
      this.pending === null &&
      this.stepCount > 0 &&
      this.stepCount % this.config.mutationInterval === 0 &&
      this.stepCount < this.config.maxSteps * 0.88;

    if (due) {
      this.attemptMutation();
    }

    if (this.stepCount % 8 === 0) this.recordHistory();
    if (this.done) {
      this.status = "done";
      this.recordFossil();
      this.recordHistory();
    }
  }

  toView(): RunView {
    const stats = parameterStats(this.genome);
    const flops = flopsPerToken(this.genome);
    const infFlops = flops * 4096;
    const cec = this.quality / ((this.trainingFlops + infFlops) / 1e15 + 0.05);
    const dac =
      this.quality /
      ((this.trainingFlops + this.searchFlops + infFlops) / 1e15 + 0.05);
    return {
      id: this.id,
      name: this.name,
      mode: this.mode,
      seed: this.config.seed,
      preset: this.preset,
      step: this.stepCount,
      maxSteps: this.config.maxSteps,
      tokens: this.tokens,
      loss: this.loss,
      valLoss: this.valLoss,
      quality: this.quality,
      cec,
      dac,
      params: stats.params,
      activeParams: stats.activeParams,
      flopsPerToken: flops,
      trainingFlops: this.trainingFlops,
      searchFlops: this.searchFlops,
      mutationCount: this.mutationCount,
      rollbackCount: this.rollbackCount,
      genome: this.genome,
      genomeHash: genomeHash(this.genome),
      fossils: this.fossils,
      mutations: this.mutations,
      history: this.history,
      importance: this.importanceView(),
      experts: this.expertView(),
      probes: { ...this.probes },
      status: this.status,
      lastMutation: this.mutations[this.mutations.length - 1] ?? null,
      exploration: this.exploration,
      architectureEntropy: this.architectureEntropy(),
    };
  }

  private sampleTaskMix(): number[] {
    const mix = TASKS.map((t) => TASK_WEIGHTS[t] * (0.7 + 0.6 * this.dataRng.next()));
    return l2normalize(mix);
  }

  private updateRouting(task: number[]): void {
    for (const layer of this.genome.layers) {
      if (!layer.enabled || layer.ffn !== "moe" || layer.experts.length === 0) continue;
      const logits = layer.experts.map((e) => {
        let s = 0;
        for (let i = 0; i < task.length; i++) s += (e.specialty[i] ?? 0) * task[i]!;
        return s * 3.4 + this.routeRng.gaussian(0, 0.15);
      });
      const p = softmax(logits);
      for (let i = 0; i < layer.experts.length; i++) {
        const id = layer.experts[i]!.id;
        const prev = this.routingEma.get(id) ?? 1 / layer.experts.length;
        this.routingEma.set(id, prev * 0.92 + p[i]! * 0.08);
      }
    }
  }

  private hebbian(task: number[]): void {
    for (const layer of this.genome.layers) {
      if (!layer.enabled || layer.ffn !== "moe") continue;
      for (const e of layer.experts) {
        const r = this.routingEma.get(e.id) ?? 0;
        if (r < 0.04) continue;
        const lr = 0.012 * r;
        e.specialty = l2normalize(
          e.specialty.map((x, i) => Math.abs(x + lr * (task[i] ?? 0))),
        );
      }
    }
  }

  private scoreArchitecture(): number {
    const enabled = this.genome.layers.filter((l) => l.enabled);
    const n = enabled.length || 1;
    const depth = this.genome.layers.length;
    let acc = 0;
    for (const layer of enabled) {
      const wantAtt = wantsAttention(layer.index, depth);
      if (wantAtt) {
        acc += layer.module === "attention" ? 1 : layer.module === "hybrid" ? 0.72 : 0.34;
        acc += layer.ffn === "dense" ? 0.22 : 0.14;
      } else {
        acc += layer.module === "gdn" ? 1 : layer.module === "hybrid" ? 0.76 : 0.42;
        acc += layer.ffn === "moe" ? 0.28 : 0.1;
      }
      if (layer.ffn === "moe") {
        const k = layer.experts.length;
        acc += 0.28 * Math.exp(-((k - 8) ** 2) / 22);
        const act = layer.activeExperts;
        acc += act === 2 ? 0.14 : act === 1 ? 0.05 : act === 3 ? 0.09 : 0.02;
        const utils = layer.experts.map((e) => this.routingEma.get(e.id) ?? 0);
        acc += 0.34 * specialization(utils);
        const dead = utils.filter((u) => u < 0.045).length;
        acc -= 0.11 * dead;
        if (utils.some((u) => u > 0.48)) acc -= 0.16;
      }
    }
    let score = acc / (n * 2.05);
    const depthFit = Math.exp(-((n - 12) ** 2) / 48);
    score *= 0.72 + 0.28 * depthFit;
    const stats = parameterStats(this.genome);
    if (stats.params > this.genome.global.maxParams) score *= 0.82;
    if (stats.activeParams > this.genome.global.maxActiveParams) score *= 0.88;
    return clamp(score, 0.06, 0.97);
  }

  private computeLoss(): number {
    const q = this.quality;
    const floor = 1.62 - 0.42 * q;
    const A = 3.05 - 0.35 * q;
    const alpha = 0.42 + 0.22 * q;
    const half = 7.2e7;
    const progress = this.tokens / (this.tokens + half);
    const decay = Math.pow(1 - progress, 1 + alpha);
    const noise = this.dataRng.gaussian(0, 0.012);
    return clamp(floor + A * decay + this.shock + noise, 1.05, 5.2);
  }

  private updateProbes(): void {
    const progress = this.stepCount / this.config.maxSteps;
    for (let i = 0; i < TASKS.length; i++) {
      const task = TASKS[i]!;
      let aff = 0;
      let n = 0;
      for (const layer of this.genome.layers) {
        if (!layer.enabled) continue;
        aff += MODULE_AFFINITY[layer.module][i] ?? 0.5;
        n += 1;
      }
      aff /= n || 1;
      const x = -1.15 + 2.35 * this.quality * aff + 1.55 * progress;
      const target = 1 / (1 + Math.exp(-x));
      this.probes[task] = this.probes[task] * 0.9 + target * 0.1;
    }
  }

  private ensureImportance(): void {
    for (const layer of this.genome.layers) {
      if (!this.importance.has(layer.id)) {
        this.importance.set(layer.id, this.freshRec(layer.id, "layer"));
      }
      for (const e of layer.experts) {
        if (!this.importance.has(e.id)) {
          this.importance.set(e.id, this.freshRec(e.id, "expert"));
        }
      }
    }
  }

  private freshRec(id: string, kind: ImportanceRec["kind"]): ImportanceRec {
    return {
      id,
      kind,
      activation: 0.3,
      routing: 0.3,
      gradient: 0.3,
      fisher: 0.3,
      uniqueness: 0.5,
      stability: 0.4,
      lifecycle: "new",
      age: 0,
      history: [],
    };
  }

  private updateImportance(): void {
    this.ensureImportance();
    for (const layer of this.genome.layers) {
      const rec = this.importance.get(layer.id);
      if (!rec) continue;
      rec.age += 1;
      const mag =
        (layer.enabled ? 0.7 : 0.05) *
        (layer.module === "attention" ? 0.85 : 0.7);
      rec.activation = rec.activation * 0.85 + mag * 0.15;
      rec.gradient = rec.gradient * 0.85 + (mag * (0.7 + 0.3 * this.quality)) * 0.15;
      rec.fisher = rec.activation * rec.gradient;
      rec.history.push(rec.activation);
      if (rec.history.length > 24) rec.history.shift();
      rec.stability = this.stabilityOf(rec.history);
      rec.lifecycle = this.lifecycleFor(rec, layer.enabled);

      if (layer.ffn !== "moe") continue;
      const specs = layer.experts.map((e) => e.specialty);
      for (const e of layer.experts) {
        const er = this.importance.get(e.id);
        if (!er) continue;
        er.age += 1;
        const r = this.routingEma.get(e.id) ?? 0;
        er.routing = er.routing * 0.8 + r * 0.2;
        er.activation = er.routing;
        er.gradient = er.gradient * 0.85 + (r * 0.8 + 0.1) * 0.15;
        er.fisher = er.activation * er.gradient;
        let uniq = 1;
        for (const s of specs) {
          if (s === e.specialty) continue;
          uniq = Math.min(uniq, 1 - Math.max(0, cosine(e.specialty, s)));
        }
        er.uniqueness = er.uniqueness * 0.85 + uniq * 0.15;
        er.history.push(er.routing);
        if (er.history.length > 24) er.history.shift();
        er.stability = this.stabilityOf(er.history);
        er.lifecycle = this.lifecycleFor(er, true);
      }
    }
  }

  private stabilityOf(hist: number[]): number {
    if (hist.length < 6) return 0.35;
    const a = hist.slice(0, Math.floor(hist.length / 2));
    const b = hist.slice(Math.floor(hist.length / 2));
    const ma = a.reduce((s, x) => s + x, 0) / a.length;
    const mb = b.reduce((s, x) => s + x, 0) / b.length;
    let num = 0;
    let da = 0;
    let db = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
      const xa = (a[i] ?? ma) - ma;
      const xb = (b[i] ?? mb) - mb;
      num += xa * xb;
      da += xa * xa;
      db += xb * xb;
    }
    const corr = num / (Math.sqrt(da * db) || 1);
    return clamp(0.5 * (corr + 1), 0, 1);
  }

  private lifecycleFor(rec: ImportanceRec, enabled: boolean): Lifecycle {
    if (!enabled) return "dying";
    const score = rec.activation * 0.5 + rec.fisher * 0.3 + rec.uniqueness * 0.2;
    if (rec.age < 4) return "new";
    if (rec.age < 10) return "emerging";
    if (score < 0.12 && rec.stability > 0.45) return "redundant";
    if (score < 0.2 && rec.stability < 0.4) return "uncertain";
    if (score > 0.72 && rec.stability > 0.55) return "essential";
    if (score > 0.45 && rec.stability > 0.5) return "stable";
    if (score < 0.16) return "dying";
    return "active";
  }

  private attemptMutation(): void {
    const snapshot = cloneGenome(this.genome);
    const result =
      this.mode === "random" ? this.randomMutation() : this.adaptiveMutation();
    if (!result || !result.ok) return;

    const safety = validateGenome(this.genome);
    if (!safety.valid) {
      this.genome = snapshot;
      this.mutations.push({
        step: this.stepCount,
        tokens: this.tokens,
        op: "rollback",
        target: result.target,
        reason: `topology safety: ${safety.errors[0] ?? "invalid genome"}`,
        accepted: true,
        lossBefore: this.loss,
        lossAfter: this.loss,
      });
      return;
    }

    this.searchFlops += flopsPerToken(this.genome) * this.config.tokensPerStep * 4;
    this.mutationCount += 1;
    const event: MutationEvent = {
      step: this.stepCount,
      tokens: this.tokens,
      op: result.op,
      target: result.target,
      reason: result.reason,
      accepted: true,
      lossBefore: this.loss,
    };
    this.shock += this.mode === "random" ? 0.16 : 0.07;
    this.pending = {
      snapshot,
      event,
      until: this.stepCount + Math.max(18, Math.floor(this.config.mutationInterval * 0.35)),
      preLoss: this.loss,
    };
    this.mutations.push(event);
    this.recordFossil();
  }

  private resolvePending(): void {
    if (!this.pending) return;
    const { snapshot, event, preLoss } = this.pending;
    const spike = this.loss > preLoss * 1.22 || this.loss > preLoss + 0.35;
    if (spike) {
      this.genome = snapshot;
      this.rollbackCount += 1;
      event.accepted = false;
      event.lossAfter = this.loss;
      this.mutations.push({
        step: this.stepCount,
        tokens: this.tokens,
        op: "rollback",
        target: event.target,
        reason: `loss spike ${this.loss.toFixed(3)} > ${preLoss.toFixed(3)}`,
        accepted: true,
        lossBefore: this.loss,
        lossAfter: preLoss,
      });
      this.shock = Math.max(0, this.shock - 0.08);
      this.recordFossil();
    } else {
      event.lossAfter = this.loss;
      this.acceptCount += 1;
    }
    this.pending = null;
    this.ensureImportance();
  }

  private randomMutation() {
    const ops: Array<"prune" | "grow" | "split" | "merge" | "rewire" | "module" | "depth"> = [
      "prune",
      "grow",
      "split",
      "merge",
      "rewire",
    ];
    if (this.config.allowModuleMutation) ops.push("module");
    if (this.config.allowDepthMutation) ops.push("depth");
    const op = this.ctrlRng.pick(ops);
    const layers = this.genome.layers.filter((l) => l.enabled);
    const moe = layers.filter((l) => l.ffn === "moe" && l.experts.length > 0);
    const layer = this.ctrlRng.pick(moe.length ? moe : layers);

    if (op === "module" && this.config.allowModuleMutation) {
      return mutateModule(layer, this.ctrlRng, "random module");
    }
    if (op === "depth" && this.config.allowDepthMutation) {
      if (this.ctrlRng.chance(0.5)) return pruneLayer(this.genome, layer, "random depth prune");
      return growLayer(this.genome, this.ctrlRng, "random depth grow");
    }
    if (layer.ffn !== "moe") return growLayer(this.genome, this.ctrlRng, "random grow");
    if (op === "prune") {
      const e = this.ctrlRng.pick(layer.experts);
      return pruneExpert(this.genome, layer, e.id, "random prune");
    }
    if (op === "grow") return growExpert(this.genome, layer, this.ctrlRng, this.stepCount, "random grow");
    if (op === "split") {
      const e = this.ctrlRng.pick(layer.experts);
      return splitExpert(this.genome, layer, e.id, this.ctrlRng, this.stepCount, "random split");
    }
    if (op === "merge" && layer.experts.length >= 3) {
      const a = this.ctrlRng.pick(layer.experts);
      const b = this.ctrlRng.pick(layer.experts.filter((e) => e.id !== a.id));
      return mergeExperts(layer, a.id, b.id, "random merge");
    }
    return rewireLayer(layer, this.ctrlRng, "random rewire");
  }

  private adaptiveMutation() {
    this.ensureImportance();
    const moeLayers = this.genome.layers.filter(
      (l) => l.enabled && l.ffn === "moe" && l.experts.length > 0,
    );

    const pruneCand: { layer: LayerGene; id: string; score: number }[] = [];
    const splitCand: { layer: LayerGene; id: string; score: number }[] = [];
    const mergeCand: { layer: LayerGene; a: string; b: string; sim: number }[] = [];

    for (const layer of moeLayers) {
      for (const e of layer.experts) {
        const rec = this.importance.get(e.id);
        if (!rec) continue;
        const load = rec.routing;
        const important = rec.activation * 0.6 + rec.fisher * 0.4;
        if (
          rec.age >= 8 &&
          rec.stability > 0.42 &&
          important < 0.16 &&
          rec.lifecycle !== "essential"
        ) {
          pruneCand.push({ layer, id: e.id, score: important });
        }
        if (rec.age >= 6 && load > 0.38 && rec.stability > 0.35) {
          splitCand.push({ layer, id: e.id, score: load });
        }
      }
      for (let i = 0; i < layer.experts.length; i++) {
        for (let j = i + 1; j < layer.experts.length; j++) {
          const a = layer.experts[i]!;
          const b = layer.experts[j]!;
          const sim = cosine(a.specialty, b.specialty);
          const ra = this.importance.get(a.id);
          const rb = this.importance.get(b.id);
          if (sim > 0.9 && (ra?.uniqueness ?? 1) < 0.18 && (rb?.uniqueness ?? 1) < 0.18) {
            mergeCand.push({ layer, a: a.id, b: b.id, sim });
          }
        }
      }
    }

    pruneCand.sort((a, b) => a.score - b.score);
    splitCand.sort((a, b) => b.score - a.score);
    mergeCand.sort((a, b) => b.sim - a.sim);

    const temp = this.exploration;
    const roll = this.ctrlRng.next();

    if (this.config.allowModuleMutation) {
      const worst = this.worstModuleLayer();
      if (worst) {
        const want = wantsAttention(worst.index, this.genome.layers.length);
        const preferred = want ? "attention" : "gdn";
        if (worst.module !== preferred && roll < 0.55 + 0.2 * temp) {
          return mutateModule(worst, this.ctrlRng, "underperforming primitive", preferred);
        }
      }
    }

    if (pruneCand.length && (roll < 0.5 || moeLayers.some((l) => l.experts.length > 9))) {
      const c = pruneCand[0]!;
      return pruneExpert(this.genome, c.layer, c.id, "low importance, stable");
    }
    if (mergeCand.length && roll < 0.68) {
      const c = mergeCand[0]!;
      return mergeExperts(c.layer, c.a, c.b, "functional redundancy");
    }
    if (splitCand.length && roll < 0.82) {
      const c = splitCand[0]!;
      if (c.layer.experts.length < 10) {
        return splitExpert(
          this.genome,
          c.layer,
          c.id,
          this.ctrlRng,
          this.stepCount,
          "routing saturation",
        );
      }
    }
    if (moeLayers.length) {
      const loaded = moeLayers.find((l) => {
        const utils = l.experts.map((e) => this.routingEma.get(e.id) ?? 0);
        return specialization(utils) > 0.45 && l.experts.length < 12;
      });
      if (loaded) {
        return growExpert(
          this.genome,
          loaded,
          this.ctrlRng,
          this.stepCount,
          "capacity limiting",
        );
      }
      const some = this.ctrlRng.pick(moeLayers);
      return rewireLayer(some, this.ctrlRng, "routing k adjust");
    }
    return growLayer(this.genome, this.ctrlRng, "grow capacity");
  }

  private worstModuleLayer(): LayerGene | null {
    const enabled = this.genome.layers.filter((l) => l.enabled);
    let worst: LayerGene | null = null;
    let worstScore = Infinity;
    const depth = this.genome.layers.length;
    for (const layer of enabled) {
      const rec = this.importance.get(layer.id);
      const want = wantsAttention(layer.index, depth);
      const match =
        want && layer.module === "attention"
          ? 1
          : !want && layer.module === "gdn"
            ? 1
            : 0.4;
      const score = (rec?.activation ?? 0.4) * match;
      if (score < worstScore) {
        worstScore = score;
        worst = layer;
      }
    }
    return worst;
  }

  private architectureEntropy(): number {
    const enabled = this.genome.layers.filter((l) => l.enabled);
    const counts = new Map<string, number>();
    for (const l of enabled) {
      const k = `${l.module}:${l.ffn}`;
      counts.set(k, (counts.get(k) ?? 0) + 1);
    }
    const n = enabled.length || 1;
    let h = 0;
    for (const c of counts.values()) {
      const p = c / n;
      h -= p * Math.log(p);
    }
    return h;
  }

  private importanceView(): ImportanceView[] {
    const out: ImportanceView[] = [];
    for (const layer of this.genome.layers) {
      const rec = this.importance.get(layer.id);
      if (rec) {
        out.push({
          id: rec.id,
          kind: "layer",
          label: `L${String(layer.index).padStart(2, "0")} ${layer.module}`,
          activation: rec.activation,
          routing: rec.routing,
          gradient: rec.gradient,
          fisher: rec.fisher,
          uniqueness: rec.uniqueness,
          stability: rec.stability,
          lifecycle: rec.lifecycle,
          age: rec.age,
        });
      }
      for (const e of layer.experts) {
        const er = this.importance.get(e.id);
        if (!er) continue;
        out.push({
          id: er.id,
          kind: "expert",
          label: `L${String(layer.index).padStart(2, "0")} ${e.id.slice(0, 8)}`,
          activation: er.activation,
          routing: er.routing,
          gradient: er.gradient,
          fisher: er.fisher,
          uniqueness: er.uniqueness,
          stability: er.stability,
          lifecycle: er.lifecycle,
          age: er.age,
        });
      }
    }
    return out;
  }

  private expertView(): ExpertView[] {
    const out: ExpertView[] = [];
    for (const layer of this.genome.layers) {
      for (const e of layer.experts) {
        const rec = this.importance.get(e.id);
        out.push({
          id: e.id,
          layerId: layer.id,
          layerIndex: layer.index,
          specialty: e.specialty,
          routing: this.routingEma.get(e.id) ?? 0,
          uniqueness: rec?.uniqueness ?? 0.5,
          lifecycle: rec?.lifecycle ?? "new",
          parentId: e.parentId,
          age: rec?.age ?? 0,
        });
      }
    }
    return out;
  }

  private recordHistory(): void {
    const stats = parameterStats(this.genome);
    const flops = flopsPerToken(this.genome);
    const cec = this.quality / (this.trainingFlops / 1e15 + 0.05);
    const last = this.history[this.history.length - 1];
    if (last && last.step === this.stepCount) {
      this.history[this.history.length - 1] = {
        step: this.stepCount,
        tokens: this.tokens,
        loss: this.loss,
        valLoss: this.valLoss,
        quality: this.quality,
        params: stats.params,
        activeParams: stats.activeParams,
        flopsPerToken: flops,
        cec,
      };
      return;
    }
    this.history.push({
      step: this.stepCount,
      tokens: this.tokens,
      loss: this.loss,
      valLoss: this.valLoss,
      quality: this.quality,
      params: stats.params,
      activeParams: stats.activeParams,
      flopsPerToken: flops,
      cec,
    });
    if (this.history.length > 360) this.history.splice(0, this.history.length - 360);
  }

  private recordFossil(): void {
    const stats = parameterStats(this.genome);
    const frame: FossilFrame = {
      step: this.stepCount,
      tokens: this.tokens,
      loss: this.loss,
      quality: this.quality,
      params: stats.params,
      activeParams: stats.activeParams,
      layers: compactLayers(this.genome) as CompactLayer[],
    };
    const last = this.fossils[this.fossils.length - 1];
    if (last && last.step === frame.step) {
      this.fossils[this.fossils.length - 1] = frame;
      return;
    }
    this.fossils.push(frame);
  }
}

export function runEngineTests(): { name: string; pass: boolean; detail: string }[] {
  const results: { name: string; pass: boolean; detail: string }[] = [];
  const g = createGenome(42, "supernet");
  results.push({
    name: "genome serializes",
    pass: JSON.parse(JSON.stringify(g)).version === 1,
    detail: genomeHash(g),
  });
  const invalid = cloneGenome(g);
  invalid.layers[0]!.index = 99;
  const validation = validateGenome(invalid);
  results.push({
    name: "topology safety rejects stale indices",
    pass: !validation.valid && validation.errors.some((e) => e.includes("stale index")),
    detail: validation.errors[0] ?? "invalid genome accepted",
  });
  const a = parameterStats(g).params;
  const e = new SimEngine({
    id: "t",
    name: "t",
    mode: "adaptive",
    seed: 7,
  });
  e.stepMany(120);
  const b = parameterStats(e.genome).params;
  results.push({
    name: "forward steps finite",
    pass: Number.isFinite(e.loss) && e.loss > 0 && e.stepCount === 120,
    detail: `loss=${e.loss.toFixed(3)} params ${a}→${b}`,
  });
  const continuous = new SimEngine({ id: "continuous", name: "continuous", mode: "adaptive", seed: 11 });
  const paused = new SimEngine({ id: "paused", name: "paused", mode: "adaptive", seed: 11 });
  continuous.stepMany(240);
  paused.stepMany(120);
  const resumed = SimEngine.fromCheckpoint(JSON.parse(JSON.stringify(paused.toCheckpoint())));
  resumed.stepMany(120);
  results.push({
    name: "checkpoint resumes deterministically",
    pass:
      genomeHash(continuous.genome) === genomeHash(resumed.genome) &&
      continuous.stepCount === resumed.stepCount &&
      Math.abs(continuous.loss - resumed.loss) < 1e-12 &&
      continuous.mutationCount === resumed.mutationCount,
    detail: `step=${resumed.stepCount} hash=${genomeHash(resumed.genome)}`,
  });
  const frozen = new SimEngine({
    id: "f",
    name: "f",
    mode: "frozen",
    seed: 3,
  });
  frozen.stepMany(200);
  results.push({
    name: "frozen never mutates",
    pass: frozen.mutationCount === 0,
    detail: `${frozen.mutationCount} mutations`,
  });
  const rnd = new SimEngine({
    id: "r",
    name: "r",
    mode: "random",
    seed: 3,
  });
  const adp = new SimEngine({
    id: "a",
    name: "a",
    mode: "adaptive",
    seed: 3,
  });
  rnd.stepMany(400);
  adp.stepMany(400);
  results.push({
    name: "same mutation cadence",
    pass: Math.abs(rnd.mutationCount - adp.mutationCount) <= 2,
    detail: `random=${rnd.mutationCount} adaptive=${adp.mutationCount}`,
  });
  results.push({
    name: "hash stable",
    pass: genomeHash(g) === genomeHash(createGenome(42, "supernet")),
    detail: genomeHash(g),
  });
  return results;
}
