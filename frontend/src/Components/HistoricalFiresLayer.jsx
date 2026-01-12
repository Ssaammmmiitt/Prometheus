import { Marker, Popup } from "react-leaflet";

const HistoricalFiresLayer = ({ fires, selectedYear }) => {
const filteredFires = fires.filter(f => f.year === selectedYear);

return (
    <>
        {filteredFires.map(fire => (
            <Marker
                key={fire.id}
                position={[fire.lat, fire.lng]}
                icon={L.divIcon({
                    className: 'fire-marker',
                    html: `<div style="width: ${Math.max(8, fire.frp / 10)}px; height: ${Math.max(8, fire.frp / 10)}px; background: rgba(239, 68, 68, 0.7); border-radius: 50%; border: 1px solid #dc2626;"></div>`,
                    iconSize: [Math.max(8, fire.frp / 10), Math.max(8, fire.frp / 10)]
                })}
            >
                <Popup>
                    <div className="text-sm">
                        <p><strong>Date:</strong> {fire.date}</p>
                        <p><strong>FRP:</strong> {fire.frp.toFixed(1)} MW</p>
                    </div>
                </Popup>
            </Marker>
        ))}
    </>
);
};

export default HistoricalFiresLayer;
