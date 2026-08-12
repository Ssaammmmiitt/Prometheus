import { Moon, Sun } from "lucide-react";

import { useTheme } from "../../theme/ThemeProvider";

export default function ThemeToggle() {
  const { isDark, toggleTheme } = useTheme();
  return (
    <button
      type="button"
      onClick={toggleTheme}
      className="min-h-11 min-w-11 flex items-center justify-center text-muted hover:text-ink transition-colors duration-200"
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
      aria-label="Toggle theme"
    >
      {isDark ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  );
}
