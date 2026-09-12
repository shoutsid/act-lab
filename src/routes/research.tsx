import { createFileRoute } from "@tanstack/react-router";
import { AppShell } from "@/components/layout/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { runEngineTests } from "@/lib/act/engine";
import { useMemo, useState } from "react";

export const Route = createFileRoute("/research")({ component: ResearchPage });

const RQS = [
  {
    id: "RQ1",
    q: "Is topology evolution actually useful?",
    a: "Does adaptive change beat a fixed architecture, random churn, and conventional pruning at matched compute?",
  },
  {
    id: "RQ2",
    q: "Can useful topology be detected before convergence?",
    a: "Does structural importance stabilize early enough to drive decisions on the slow clock?",
  },
  {
    id: "RQ3",
    q: "Which importance signals work?",
    a: "Magnitude, routing, gradient, Fisher, ablation, uniqueness, temporal stability.",
  },
  {
    id: "RQ4",
    q: "Does causal structural selection matter?",
    a: "Importance-guided evolution versus identical random topology churn — Experiment 0.",
  },
  {
    id: "RQ5",
    q: "Are discovered structures reusable?",
    a: "Extract the genome, reinitialize, train independently.",
  },
  {
    id: "RQ6",
    q: "Initialization or topology?",
    a: "Evolved weights vs rewind vs fresh vs teacher.",
  },
  {
    id: "RQ7",
    q: "Does architecture scale?",
    a: "A 300M discovery transferred to 500M / 1B / 3B.",
  },
  {
    id: "RQ8",
    q: "Competitive with Qwen-like hybrids?",
    a: "Uninformed search versus Qwen-seeded priors, never Qwen weights in V0.",
  },
];

const GATES = [
  { id: "1", t: "Dense baseline trains and resumes" },
  { id: "2", t: "Random churn is reproducible" },
  { id: "3", t: "Adaptive > random", note: "If failed: stop and simplify" },
  { id: "4", t: "Adaptive > static MoE" },
  { id: "5", t: "Fresh architecture remains competitive" },
  { id: "6", t: "Transfers across seeds" },
  { id: "7", t: "Transfers across scale" },
];

const SPRINTS = [
  {
    n: "Sprint 1 — this console",
    items: [
      "Dense / MoE genomes",
      "Topology graph serialize",
      "Prune, grow, split, merge, rewire",
      "Importance records + stability",
      "Random vs adaptive controllers",
      "Experiment 0 protocol",
      "Checkpoint-shaped fossils",
    ],
  },
  {
    n: "Sprint 2",
    items: ["Fisher", "Functional similarity", "Temporary ablation", "Expert ecology graphs"],
  },
  {
    n: "Sprint 3",
    items: ["Gated DeltaNet as a real kernel", "Module-type mutation", "Adaptive depth"],
  },
  {
    n: "Sprint 4–6",
    items: [
      "Qwen architecture priors",
      "Fresh / rewind / teacher init",
      "Scale 500M–1B on T4×2 / TPU",
    ],
  },
];

export function ResearchPage() {
  const [tests, setTests] = useState<ReturnType<typeof runEngineTests> | null>(null);
  const passed = useMemo(
    () => tests?.every((t) => t.pass) ?? null,
    [tests],
  );

  return (
    <AppShell>
      <main className="px-4 py-8 sm:px-6">
        <article className="mx-auto max-w-3xl">
          <p className="font-mono text-[0.6875rem] uppercase tracking-[0.18em] text-muted">
            Protocol · September 2026
          </p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">
            What this project is testing
          </h1>
          <p className="mt-4 text-base leading-relaxed text-muted">
            Not a frontier pretrain. The Kaggle-free phase exists to prove whether
            architecture discovery is a useful extra degree of freedom in
            language-model training. The valuable artifact is the genome and the
            controller — not a single checkpoint.
          </p>

          <h2 className="mt-12 text-xl font-medium">The core scientific test</h2>
          <blockquote className="mt-3 border-l border-accent/40 pl-4 text-base leading-relaxed">
            Given the same data, compute budget, parameter budget, and
            initialization, can a topology discovered through training outperform
            a topology designed by humans — and remain advantageous when freshly
            initialized?
          </blockquote>

          <h2 className="mt-12 text-xl font-medium">Research questions</h2>
          <ol className="mt-4 space-y-4">
            {RQS.map((r) => (
              <li key={r.id} className="rounded-lg bg-surface p-4 shadow-[var(--shadow-border)]">
                <div className="flex items-baseline gap-2">
                  <Badge>{r.id}</Badge>
                  <h3 className="text-sm font-medium">{r.q}</h3>
                </div>
                <p className="mt-2 text-sm leading-relaxed text-muted">{r.a}</p>
              </li>
            ))}
          </ol>

          <h2 className="mt-12 text-xl font-medium">Go / no-go gates</h2>
          <ol className="mt-4 space-y-2">
            {GATES.map((g) => (
              <li
                key={g.id}
                className="flex gap-3 rounded-md bg-surface px-3 py-3 shadow-[var(--shadow-border)]"
              >
                <span className="font-mono text-xs text-muted">{g.id}</span>
                <span>
                  <span className="text-sm">{g.t}</span>
                  {g.note ? (
                    <span className="mt-0.5 block text-xs text-warn">{g.note}</span>
                  ) : null}
                </span>
              </li>
            ))}
          </ol>

          <h2 className="mt-12 text-xl font-medium">Implementation sprints</h2>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            {SPRINTS.map((s) => (
              <section
                key={s.n}
                className="rounded-lg bg-surface p-4 shadow-[var(--shadow-border)]"
              >
                <h3 className="text-sm font-medium">{s.n}</h3>
                <ul className="mt-2 space-y-1 text-sm text-muted">
                  {s.items.map((it) => (
                    <li key={it}>{it}</li>
                  ))}
                </ul>
              </section>
            ))}
          </div>

          <h2 className="mt-12 text-xl font-medium">What this console is</h2>
          <p className="mt-3 text-sm leading-relaxed text-muted">
            A laboratory-scale implementation of the V0 engine: genomes, discrete
            operators, importance, a rule-based controller, Experiment 0, fossils,
            and the ledger. Training dynamics are a calibrated simulator with a
            hidden architectural attractor — not a 256M PyTorch run. That is
            intentional. The first objective is to demonstrate the mechanism
            before spending T4×2 sessions on real tokens.
          </p>
          <p className="mt-3 text-sm leading-relaxed text-muted">
            Next hardware step: port the frozen genome to PyTorch, train the dense
            baseline on T4×2, then repeat Experiment 0 with real gradients.
          </p>

          <h2 className="mt-12 text-xl font-medium">Engine invariants</h2>
          <p className="mt-3 text-sm text-muted">
            Tiny deterministic checks for serialization, finite loss, frozen
            mutation count, and matched churn cadence.
          </p>
          <Button
            variant="outline"
            className="mt-4"
            onClick={() => setTests(runEngineTests())}
          >
            Run engine tests
          </Button>
          {tests ? (
            <ul className="mt-4 space-y-2">
              {tests.map((t) => (
                <li
                  key={t.name}
                  className="flex flex-col rounded-md bg-surface px-3 py-2 text-sm shadow-[var(--shadow-border)] sm:flex-row sm:items-center sm:justify-between"
                >
                  <span>
                    <Badge tone={t.pass ? "good" : "warn"}>{t.pass ? "pass" : "fail"}</Badge>
                    <span className="ml-2">{t.name}</span>
                  </span>
                  <span className="mt-1 font-mono text-xs text-muted sm:mt-0">
                    {t.detail}
                  </span>
                </li>
              ))}
              <li className="text-xs text-muted">
                {passed ? "All invariants held." : "One or more invariants failed."}
              </li>
            </ul>
          ) : null}

          <p className="mt-16 text-xs leading-relaxed text-subtle">
            Do not let the project become an extremely complicated way to prune a
            model. The progression is pruning → dynamic sparsity → topology
            evolution → architecture discovery → reusability → computational
            genome. Stop adding mechanisms the moment they stop improving the
            capability/FLOP frontier.
          </p>
        </article>
      </main>
    </AppShell>
  );
}
