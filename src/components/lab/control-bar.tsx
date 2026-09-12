import { Button } from "@/components/ui/button";
import { useLab, type Speed } from "@/lib/act/store";
import { Pause, Play, RotateCcw, Save } from "lucide-react";
import { toast } from "sonner";

export function ControlBar() {
  const playing = useLab((s) => s.playing);
  const speed = useLab((s) => s.speed);
  const runs = useLab((s) => s.runs);
  const selectedId = useLab((s) => s.selectedId);
  const togglePlay = useLab((s) => s.togglePlay);
  const setSpeed = useLab((s) => s.setSpeed);
  const select = useLab((s) => s.select);
  const reset = useLab((s) => s.reset);
  const saveCurrent = useLab((s) => s.saveCurrent);
  const allDone = runs.length > 0 && runs.every((r) => r.status === "done");

  return (
    <div className="flex flex-wrap items-center gap-2">
      {runs.length > 1
        ? runs.map((r) => (
            <button
              key={r.id}
              type="button"
              onClick={() => select(r.id)}
              className={`h-11 rounded-sm px-3 font-mono text-xs transition-colors duration-150 ${
                selectedId === r.id
                  ? "bg-accent text-accent-fg"
                  : "bg-surface text-muted hover:text-fg"
              }`}
            >
              {r.name}
            </button>
          ))
        : null}
      <Button
        variant="outline"
        size="icon"
        onClick={togglePlay}
        disabled={allDone}
        aria-label={playing ? "Pause" : "Train"}
      >
        {playing ? <Pause /> : <Play />}
      </Button>
      {([1, 4, 16] as Speed[]).map((s) => (
        <button
          key={s}
          type="button"
          onClick={() => setSpeed(s)}
          className={`h-11 min-w-11 rounded-sm px-2 font-mono text-xs ${
            speed === s ? "bg-surface-2 text-fg" : "text-muted hover:text-fg"
          }`}
        >
          {s}×
        </button>
      ))}
      <Button
        variant="ghost"
        size="icon"
        onClick={() => {
          saveCurrent();
          toast("Run written to the experiment ledger");
        }}
        aria-label="Save to ledger"
      >
        <Save />
      </Button>
      <Button variant="ghost" size="icon" onClick={reset} aria-label="Reset">
        <RotateCcw />
      </Button>
    </div>
  );
}
