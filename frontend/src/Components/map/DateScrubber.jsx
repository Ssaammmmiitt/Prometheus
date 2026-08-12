import { useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play } from "lucide-react";

import { datesForYear } from "../../lib/nepal";
import { prettyDate } from "../../lib/plain";
import Button from "../ui/Button";
import Card from "../ui/Card";
import Tabs from "../ui/Tabs";

const MIN_MS = 450;
const MAX_MS = 1200;
const DEFAULT_MS = 750;

function sliderFromMs(ms) {
  return Math.round(((MAX_MS - ms) / (MAX_MS - MIN_MS)) * 100);
}

function msFromSlider(value) {
  return Math.round(MAX_MS - (value / 100) * (MAX_MS - MIN_MS));
}

export default function DateScrubber({
  date,
  dates,
  years,
  onDate,
  playing,
  onPlaying,
}) {
  const year = date.slice(0, 4);
  const seasonDates = useMemo(() => datesForYear(dates, year), [dates, year]);
  const index = Math.max(0, seasonDates.indexOf(date));
  const timer = useRef(null);
  const dateRef = useRef(date);
  dateRef.current = date;
  const [speed, setSpeed] = useState(() => sliderFromMs(DEFAULT_MS));
  const delay = msFromSlider(speed);

  useEffect(() => {
    if (!playing) {
      if (timer.current) clearInterval(timer.current);
      return undefined;
    }
    timer.current = setInterval(() => {
      if (!seasonDates.length) return;
      const i = seasonDates.indexOf(dateRef.current);
      const next = seasonDates[(i + 1) % seasonDates.length];
      onDate(next, { live: true });
    }, delay);
    return () => clearInterval(timer.current);
  }, [playing, delay, seasonDates, onDate]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.code !== "Space") return;
      const tag = e.target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      e.preventDefault();
      if (playing) {
        onDate(dateRef.current);
        onPlaying(false);
      } else {
        onPlaying(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [playing, onPlaying, onDate]);

  const togglePlay = () => {
    if (playing) {
      onDate(date);
      onPlaying(false);
    } else {
      onPlaying(true);
    }
  };

  return (
    <Card className="absolute bottom-3 md:bottom-4 left-1/2 -translate-x-1/2 z-900 w-[min(720px,calc(100vw-1rem))] p-3 md:p-4 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
      <div className="flex items-center gap-2 md:gap-3 mb-2 md:mb-3">
        <Tabs
          value={year}
          onChange={(y) => {
            const next = datesForYear(dates, y);
            onDate(next.includes(date) ? date : next[Math.floor(next.length / 2)] || date);
            onPlaying(false);
          }}
          options={(years.length ? years : ["2024", "2025", "2026"]).map((y) => ({
            value: y,
            label: y,
          }))}
        />
        <span
          key={date}
          className="flex-1 text-center font-display font-extrabold text-xl md:text-2xl tracking-tight tabular-nums panel-enter min-w-0 truncate"
        >
          {prettyDate(date)}
        </span>
        <label className="flex flex-col justify-center shrink-0 w-[25%] max-w-[11rem] min-w-[5.5rem]">
          <span className="label-ui text-muted leading-none mb-1">Speed</span>
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="w-full accent-[var(--accent)] range-compact"
            aria-label="Playback speed"
          />
        </label>
        <Button
          variant={playing ? "outline" : "primary"}
          className={`px-3 md:px-[22px] min-h-11 ${playing ? "playing-cta" : ""}`}
          onClick={togglePlay}
        >
          {playing ? <Pause size={14} /> : <Play size={14} />}
          <span className="hidden sm:inline">{playing ? "Pause" : "Play season"}</span>
        </Button>
      </div>
      <input
        type="range"
        min={0}
        max={Math.max(0, seasonDates.length - 1)}
        value={index}
        onChange={(e) => {
          const d = seasonDates[Number(e.target.value)];
          if (d) onDate(d);
        }}
        className="w-full accent-[var(--accent)]"
        aria-label="Move through the fire season"
      />
      <div className="flex justify-between mt-1 gap-2">
        <span className="label-ui text-muted truncate">
          {prettyDate(seasonDates[0])}
        </span>
        <span className="label-ui text-muted hidden sm:inline">
          Drag or tap Play to watch the season
        </span>
        <span className="label-ui text-muted truncate">
          {prettyDate(seasonDates[seasonDates.length - 1])}
        </span>
      </div>
    </Card>
  );
}
