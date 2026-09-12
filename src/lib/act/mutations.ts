import {
  type ArchitectureGenome,
  type ExpertGene,
  type LayerGene,
  type ModuleType,
  type MutationOp,
  clamp,
  createLayer,
  l2normalize,
  parameterStats,
  reindex,
  uid,
} from "./genome";
import { type Rng } from "./rng";

export interface MutationResult {
  op: MutationOp;
  target: string;
  reason: string;
  ok: boolean;
}

function withinBudget(g: ArchitectureGenome): boolean {
  const s = parameterStats(g);
  return (
    s.params <= g.global.maxParams &&
    s.activeParams <= g.global.maxActiveParams
  );
}

export function pruneExpert(
  g: ArchitectureGenome,
  layer: LayerGene,
  expertId: string,
  reason: string,
): MutationResult {
  if (layer.experts.length <= 2) {
    return { op: "prune", target: expertId, reason: "min experts", ok: false };
  }
  const idx = layer.experts.findIndex((e) => e.id === expertId);
  if (idx < 0) return { op: "prune", target: expertId, reason, ok: false };
  layer.experts.splice(idx, 1);
  layer.activeExperts = Math.min(layer.activeExperts, layer.experts.length);
  return { op: "prune", target: expertId, reason, ok: true };
}

export function growExpert(
  g: ArchitectureGenome,
  layer: LayerGene,
  rng: Rng,
  step: number,
  reason: string,
): MutationResult {
  if (layer.ffn !== "moe") {
    return { op: "grow", target: layer.id, reason: "not moe", ok: false };
  }
  if (layer.experts.length >= 16) {
    return { op: "grow", target: layer.id, reason: "max experts", ok: false };
  }
  const width = layer.experts[0]?.width ?? layer.ffnWidth;
  const donor = layer.experts[rng.int(layer.experts.length)];
  const spec = l2normalize(
    (donor?.specialty ?? Array(6).fill(1)).map(
      (x) => x + rng.gaussian(0, 0.18),
    ),
  );
  const expert: ExpertGene = {
    id: uid("e", rng),
    parentId: donor?.id ?? null,
    birthStep: step,
    specialty: spec.map((x) => Math.abs(x)),
    width,
  };
  expert.specialty = l2normalize(expert.specialty.map((x) => Math.abs(x)));
  layer.experts.push(expert);
  if (!withinBudget(g)) {
    layer.experts.pop();
    return { op: "grow", target: layer.id, reason: "budget", ok: false };
  }
  return { op: "grow", target: expert.id, reason, ok: true };
}

export function splitExpert(
  g: ArchitectureGenome,
  layer: LayerGene,
  expertId: string,
  rng: Rng,
  step: number,
  reason: string,
): MutationResult {
  const src = layer.experts.find((e) => e.id === expertId);
  if (!src || layer.experts.length >= 16) {
    return { op: "split", target: expertId, reason: "cannot split", ok: false };
  }
  const eps = 0.12;
  const a = l2normalize(src.specialty.map((x) => Math.abs(x + rng.gaussian(0, eps))));
  const b = l2normalize(src.specialty.map((x) => Math.abs(x - rng.gaussian(0, eps))));
  const child: ExpertGene = {
    id: uid("e", rng),
    parentId: src.id,
    birthStep: step,
    specialty: b,
    width: src.width,
  };
  src.specialty = a;
  layer.experts.push(child);
  if (!withinBudget(g)) {
    layer.experts.pop();
    src.specialty = l2normalize(src.specialty);
    return { op: "split", target: expertId, reason: "budget", ok: false };
  }
  return { op: "split", target: `${src.id}→${child.id}`, reason, ok: true };
}

export function mergeExperts(
  layer: LayerGene,
  aId: string,
  bId: string,
  reason: string,
): MutationResult {
  const ia = layer.experts.findIndex((e) => e.id === aId);
  const ib = layer.experts.findIndex((e) => e.id === bId);
  if (ia < 0 || ib < 0 || ia === ib || layer.experts.length <= 2) {
    return { op: "merge", target: `${aId}+${bId}`, reason: "cannot merge", ok: false };
  }
  const lo = Math.min(ia, ib);
  const hi = Math.max(ia, ib);
  const survivor = layer.experts[lo]!;
  const gone = layer.experts[hi]!;
  survivor.specialty = l2normalize(
    survivor.specialty.map((x, i) => 0.5 * x + 0.5 * (gone.specialty[i] ?? 0)),
  );
  layer.experts.splice(hi, 1);
  layer.activeExperts = Math.min(layer.activeExperts, layer.experts.length);
  return {
    op: "merge",
    target: `${survivor.id}+${gone.id}`,
    reason,
    ok: true,
  };
}

export function rewireLayer(
  layer: LayerGene,
  rng: Rng,
  reason: string,
): MutationResult {
  if (layer.ffn !== "moe" || layer.experts.length < 2) {
    return { op: "rewire", target: layer.id, reason: "not moe", ok: false };
  }
  const maxK = Math.min(4, layer.experts.length);
  const next = clamp(layer.activeExperts + (rng.chance(0.5) ? 1 : -1), 1, maxK);
  if (next === layer.activeExperts) {
    return { op: "rewire", target: layer.id, reason: "no change", ok: false };
  }
  layer.activeExperts = next;
  return {
    op: "rewire",
    target: layer.id,
    reason: `${reason} (k=${next})`,
    ok: true,
  };
}

const MODULES: ModuleType[] = ["attention", "gdn", "hybrid"];

export function mutateModule(
  layer: LayerGene,
  rng: Rng,
  reason: string,
  preferred?: ModuleType,
): MutationResult {
  const next =
    preferred ??
    rng.pick(MODULES.filter((m) => m !== layer.module));
  if (next === layer.module) {
    return { op: "module", target: layer.id, reason: "same", ok: false };
  }
  const prev = layer.module;
  layer.module = next;
  return {
    op: "module",
    target: layer.id,
    reason: `${reason} (${prev}→${next})`,
    ok: true,
  };
}

export function pruneLayer(
  g: ArchitectureGenome,
  layer: LayerGene,
  reason: string,
): MutationResult {
  const enabled = g.layers.filter((l) => l.enabled).length;
  if (enabled <= 8) {
    return { op: "prune", target: layer.id, reason: "min depth", ok: false };
  }
  layer.enabled = false;
  return { op: "prune", target: layer.id, reason, ok: true };
}

export function growLayer(
  g: ArchitectureGenome,
  rng: Rng,
  reason: string,
): MutationResult {
  const disabled = g.layers.find((l) => !l.enabled);
  if (disabled) {
    disabled.enabled = true;
    return { op: "grow", target: disabled.id, reason, ok: true };
  }
  if (g.layers.length >= 16) {
    return { op: "grow", target: "depth", reason: "max depth", ok: false };
  }
  const donor = rng.pick(g.layers.filter((l) => l.enabled));
  const layer = createLayer(rng, g.layers.length, {
    module: donor.module,
    ffn: donor.ffn,
    hiddenSize: g.hiddenSize,
    heads: donor.heads,
    kvHeads: donor.kvHeads,
    ffnWidth: donor.ffnWidth,
    expertCount: donor.ffn === "moe" ? Math.min(8, donor.experts.length || 8) : 0,
    activeExperts: donor.activeExperts,
    sharedExpert: donor.sharedExpert,
    expertWidth: donor.experts[0]?.width ?? g.hiddenSize,
  });
  g.layers.push(layer);
  reindex(g);
  if (!withinBudget(g)) {
    g.layers.pop();
    reindex(g);
    return { op: "grow", target: "depth", reason: "budget", ok: false };
  }
  return { op: "grow", target: layer.id, reason, ok: true };
}

export function growHeads(
  g: ArchitectureGenome,
  layer: LayerGene,
  reason: string,
): MutationResult {
  if (layer.module === "gdn") {
    return { op: "grow", target: layer.id, reason: "gdn has no heads", ok: false };
  }
  if (layer.heads >= 16) {
    return { op: "grow", target: layer.id, reason: "max heads", ok: false };
  }
  layer.heads = Math.min(16, layer.heads + 2);
  layer.kvHeads = Math.max(1, Math.floor(layer.heads / 4));
  if (!withinBudget(g)) {
    layer.heads -= 2;
    layer.kvHeads = Math.max(1, Math.floor(layer.heads / 4));
    return { op: "grow", target: layer.id, reason: "budget", ok: false };
  }
  return { op: "grow", target: `${layer.id}:heads=${layer.heads}`, reason, ok: true };
}

export function pruneHeads(layer: LayerGene, reason: string): MutationResult {
  if (layer.module === "gdn") {
    return { op: "prune", target: layer.id, reason: "gdn has no heads", ok: false };
  }
  if (layer.heads <= 4) {
    return { op: "prune", target: layer.id, reason: "min heads", ok: false };
  }
  layer.heads -= 2;
  layer.kvHeads = Math.max(1, Math.floor(layer.heads / 4));
  return { op: "prune", target: `${layer.id}:heads=${layer.heads}`, reason, ok: true };
}
