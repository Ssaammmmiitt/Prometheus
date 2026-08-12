import { NavLink } from "react-router-dom";

import { useForecast } from "../../state/ForecastContext";
import { prettyDate } from "../../lib/plain";
import ThemeToggle from "./ThemeToggle";

const LINKS = [
  { to: "/", label: "Map", hint: "Where is it dangerous?", end: true },
  { to: "/fires", label: "Fires", hint: "Where fires already were" },
  { to: "/verify", label: "Accuracy", hint: "Did yesterday's map work?" },
];

export default function AppHeader() {
  const { query, date } = useForecast();

  return (
    <header className="absolute top-0 inset-x-0 z-1000 border-b border-[var(--hairline)] bg-surface/95 backdrop-blur-sm pt-[env(safe-area-inset-top)]">
      <div className="h-14 flex items-center justify-between gap-3 px-3 md:px-4">
        <div className="flex items-center gap-3 md:gap-6 min-w-0">
          <NavLink to={`/?${query}`} className="shrink-0">
            <span className="font-display font-extrabold text-lg md:text-[1.35rem] tracking-[-0.015em] text-ink">
              PROMETHEUS
            </span>
          </NavLink>
          <nav className="flex items-end gap-3 md:gap-5 overflow-x-auto">
            {LINKS.map((link) => (
              <NavLink
                key={link.to}
                to={`${link.to}?${query}`}
                end={link.end}
                title={link.hint}
                className={({ isActive }) =>
                  `label-ui pb-1 border-b-2 whitespace-nowrap transition-colors duration-200 ${
                    isActive
                      ? "text-accent border-accent"
                      : "text-muted border-transparent hover:text-ink"
                  }`
                }
              >
                {link.label}
              </NavLink>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <p className="hidden lg:block text-[11px] text-muted tracking-wide">
            Nepal fire chance · {prettyDate(date)}
          </p>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
