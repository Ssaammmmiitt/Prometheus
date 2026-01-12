
// const HistoricalFiresLayer = ({ fires, selectedYear }) => {
// const filteredFires = fires.filter(f => f.year === selectedYear);

// return (
//     <>
//         {filteredFires.map(fire => (
//             <Marker
//                 key={fire.id}
//                 position={[fire.lat, fire.lng]}
//                 icon={L.divIcon({
//                     className: 'fire-marker',
//                     html: `<div style="width: ${Math.max(8, fire.frp / 10)}px; height: ${Math.max(8, fire.frp / 10)}px; background: rgba(239, 68, 68, 0.7); border-radius: 50%; border: 1px solid #dc2626;"></div>`,
//                     iconSize: [Math.max(8, fire.frp / 10), Math.max(8, fire.frp / 10)]
//                 })}
//             >
//                 <Popup>
//                     <div className="text-sm">
//                         <p><strong>Date:</strong> {fire.date}</p>
//                         <p><strong>FRP:</strong> {fire.frp.toFixed(1)} MW</p>
//                     </div>
//                 </Popup>
//             </Marker>
//         ))}
//     </>
// );
// };

// export default HistoricalFiresLayer;


import React from 'react';
import { CircleMarker, Popup } from "react-leaflet";

const HistoricalFiresLayer = ({ fires, selectedYear }) => {
    // Filter fires for the selected year
    const filteredFires = React.useMemo(() => {
        return fires.filter(f => f.year === selectedYear);
    }, [fires, selectedYear]);

    return (
        <>
            {filteredFires.map(fire => (
                <CircleMarker
                    key={fire.id}
                    center={[fire.lat, fire.lng]}
                    // Use 'frp' if that is what your generator produces, or fallback to 10
                    radius={Math.max(4, (fire.frp || fire.conf || 40) / 20)}
                    pathOptions={{
                        fillColor: '#ef4444',
                        color: '#dc2626',
                        weight: 1,
                        opacity: 0.8,
                        fillOpacity: 0.4
                    }}
                >
                    <Popup>
                        <div className="text-sm p-1">
                            <p className="font-bold text-red-600 mb-1 leading-none">Fire Incident ({fire.year})</p>
                            <div className="space-y-1 mt-2 text-slate-700">
                                <p><strong>Date:</strong> {fire.date || 'N/A'}</p>
                                <p><strong>Intensity (FRP):</strong> {(fire.frp || 0).toFixed(1)}</p>
                                <p className="text-[10px] font-mono text-slate-400">
                                    {fire.lat.toFixed(4)}°N, {fire.lng.toFixed(4)}°E
                                </p>
                            </div>
                        </div>
                    </Popup>
                </CircleMarker>
            ))}
        </>
    );
};

export default HistoricalFiresLayer;