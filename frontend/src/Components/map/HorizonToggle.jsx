import Tabs from "../ui/Tabs";

export default function HorizonToggle({ value, onChange, compact = false }) {
  return (
    <Tabs
      value={value}
      onChange={onChange}
      options={[
        { value: 1, label: compact ? "1 day" : "Tomorrow" },
        { value: 7, label: compact ? "7 days" : "Next 7 days" },
      ]}
    />
  );
}
