import { useEffect, useState } from "react";

export default function AnimatedNumber({ value }) {
  const [displayValue, setDisplayValue] = useState(value);
  const [pulse, setPulse] = useState(false);

  useEffect(() => {
    if (value !== displayValue) {
      setPulse(true);
      setDisplayValue(value);
      const t = setTimeout(() => setPulse(false), 300);
      return () => clearTimeout(t);
    }
  }, [value, displayValue]);

  return (
    <span
      className={`transition-colors duration-300 ${
        pulse ? "text-accent" : "text-inherit"
      }`}
    >
      {displayValue}
    </span>
  );
}
