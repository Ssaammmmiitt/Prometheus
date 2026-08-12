import { cn } from "../../lib/cn";

export default function Card({ className, featured = false, children, ...props }) {
  return (
    <div
      className={cn(
        "bg-lift text-ink border border-[var(--hairline)] p-5",
        featured && "rounded-lg",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}
