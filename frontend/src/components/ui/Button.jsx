import { cn } from "../../lib/cn";

const variants = {
  primary:
    "bg-accent text-accent-fg border-0 font-bold hover:brightness-95",
  secondary:
    "bg-transparent text-ink border border-[var(--hairline-strong)] font-semibold hover:bg-[var(--accent-12)]",
  outline:
    "bg-transparent text-accent border border-accent font-semibold hover:bg-[var(--accent-12)]",
  ghost: "bg-transparent text-muted border-0 font-semibold hover:text-ink",
};

export default function Button({
  variant = "primary",
  className,
  children,
  ...props
}) {
  return (
    <button
      type="button"
      className={cn(
        "inline-flex items-center justify-center gap-2 min-h-11 px-4 md:px-[22px] py-3 text-[0.8125rem] tracking-[0.1em] uppercase transition-colors duration-200 disabled:opacity-40",
        variants[variant],
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}
