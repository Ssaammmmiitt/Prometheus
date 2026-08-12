import Tabs from "../ui/Tabs";

export default function HorizonToggle({ value, onChange }) {
  return (
    <Tabs
      value={value}
      onChange={onChange}
      options={[
        { value: 1, label: "Tomorrow" },
        { value: 7, label: "Next 7 days" },
      ]}
    />
  );
}
