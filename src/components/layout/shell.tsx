import { Link, useRouterState } from "@tanstack/react-router";
import { ActMark } from "@/components/mark";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

const NAV = [
  { to: "/", label: "Lab" },
  { to: "/experiments", label: "Ledger" },
  { to: "/research", label: "Research" },
] as const;

export function AppShell({
  children,
  aside,
}: {
  children: ReactNode;
  aside?: ReactNode;
}) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  return (
    <div className="min-h-dvh text-fg">
      <header className="sticky top-0 z-30 border-b border-border bg-bg/85 backdrop-blur-sm">
        <div className="mx-auto flex max-w-[1440px] items-center gap-4 px-4 py-3 sm:px-6">
          <Link to="/" className="flex min-h-11 items-center gap-2.5">
            <ActMark className="size-7" />
            <span className="flex flex-col leading-none">
              <span className="text-sm font-semibold tracking-tight">ACT Lab</span>
              <span className="hidden text-[0.625rem] uppercase tracking-[0.16em] text-muted sm:block">
                Topology
              </span>
            </span>
          </Link>
          <nav className="ml-auto flex items-center gap-1 sm:ml-8 sm:mr-auto">
            {NAV.map((item) => {
              const active =
                item.to === "/"
                  ? pathname === "/"
                  : pathname.startsWith(item.to);
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  className={cn(
                    "inline-flex h-11 items-center rounded-sm px-3 text-sm transition-colors duration-150",
                    active ? "text-fg" : "text-muted hover:text-fg",
                  )}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
          {aside}
        </div>
      </header>
      <div className="mx-auto max-w-[1440px]">{children}</div>
    </div>
  );
}
