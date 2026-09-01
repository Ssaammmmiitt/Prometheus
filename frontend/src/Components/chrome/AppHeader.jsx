import { NavLink } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Menu, X } from "lucide-react";
import { useState } from "react";

import { useForecast } from "../../state/ForecastContext";
import { prettyDate } from "../../lib/plain";
import ThemeToggle from "./ThemeToggle";
import LanguageToggle from "./LanguageToggle";

const LINKS = [
  { to: "/", labelKey: "nav.map", hint: "Where is it dangerous?", end: true },
  { to: "/compare", labelKey: "nav.compare", hint: "Compare two dates" },
  { to: "/predict", labelKey: "nav.whatif", hint: "Change the weather at a place" },
  { to: "/fires", labelKey: "nav.fires", hint: "Where fires already were" },
  { to: "/verify", labelKey: "nav.accuracy", hint: "Did yesterday's map work?" },
];

export default function AppHeader() {
  const { query, date } = useForecast();
  const { t } = useTranslation();
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header className="absolute top-0 inset-x-0 z-1000 border-b border-[var(--hairline)] bg-surface/95 backdrop-blur-sm pt-[env(safe-area-inset-top,0px)]">
      <div className="h-14 flex items-center justify-between gap-2 sm:gap-3 px-2 sm:px-3 md:px-4">
        <div className="flex items-center gap-2 sm:gap-4 md:gap-6 min-w-0 flex-1">
          <NavLink to={`/?${query}`} className="shrink-0" onClick={() => setMenuOpen(false)}>
            <span className="font-display font-extrabold text-base sm:text-lg md:text-[1.35rem] tracking-[-0.015em] text-ink">
              PROMETHEUS
            </span>
          </NavLink>
          
          {/* Desktop Nav */}
          <nav className="hidden md:flex items-end gap-5 overflow-x-auto no-scrollbar min-w-0">
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
                {t(link.labelKey)}
              </NavLink>
            ))}
          </nav>
        </div>

        <div className="flex items-center gap-1 sm:gap-2 shrink-0">
          <p className="hidden lg:block text-[11px] text-muted tracking-wide max-w-[14rem] truncate">
            {t("header.subtitle", { date: prettyDate(date) })}
          </p>
          <LanguageToggle />
          <ThemeToggle />
          
          {/* Mobile Menu Toggle */}
          <button 
            type="button" 
            className="md:hidden p-1.5 text-ink rounded-md hover:bg-[var(--accent-12)]"
            onClick={() => setMenuOpen(!menuOpen)}
            aria-label="Toggle Menu"
          >
            {menuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </div>

      {/* Mobile Nav Drawer */}
      {menuOpen && (
        <nav className="md:hidden absolute top-14 left-0 w-full bg-surface border-b border-[var(--hairline)] flex flex-col p-4 shadow-lg shadow-black/5 gap-4">
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={`${link.to}?${query}`}
              end={link.end}
              onClick={() => setMenuOpen(false)}
              className={({ isActive }) =>
                `text-base font-medium py-2 px-3 rounded-md transition-colors ${
                  isActive
                    ? "bg-[var(--accent-12)] text-accent"
                    : "text-ink hover:bg-[var(--hairline)]"
                }`
              }
            >
              {t(link.labelKey)}
            </NavLink>
          ))}
        </nav>
      )}
    </header>
  );
}
