import { cn } from "@/lib/utils";
import type { HTMLAttributes } from "react";

export function Badge({
  className,
  tone = "muted",
  ...props
}: HTMLAttributes<HTMLSpanElement> & {
  tone?: "muted" | "good" | "warn" | "att" | "gdn" | "moe" | "accent";
}) {
  const tones: Record<string, string> = {
    muted: "text-muted bg-surface-2",
    good: "text-good bg-good/10",
    warn: "text-warn bg-warn/10",
    att: "text-att bg-att/10",
    gdn: "text-gdn bg-gdn/10",
    moe: "text-moe bg-moe/10",
    accent: "text-accent-fg bg-accent",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 font-mono text-[0.6875rem] tracking-wide uppercase",
        tones[tone],
        className,
      )}
      {...props}
    />
  );
}
