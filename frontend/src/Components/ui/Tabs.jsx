import { cn } from "../../lib/cn";

export default function Tabs({ options, value, onChange, className }) {
  return (
    <div className={cn("flex items-end gap-3 md:gap-5 overflow-x-auto", className)}>
      {options.map((opt) => {
        const id = String(opt.value);
        const active = String(value) === id;
        return (
          <button
            key={id}
            type="button"
            onClick={() => onChange(opt.value)}
            className={cn(
              "label-ui min-h-11 shrink-0 pb-1 border-b-2 transition-colors duration-200",
              active
                ? "text-accent border-accent"
                : "text-muted border-transparent hover:text-ink",
            )}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
