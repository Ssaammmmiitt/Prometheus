"use client";
import React from 'react';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer } from 'recharts';

const SidePanel = ({ patch, onClose, useTheme, threshold, modelVersion, getProbabilityColor }) => {
    const { isDark } = useTheme();

    if (!patch) return null;

    const prob = patch.probability ?? 0;
    const sensors = patch.sensor_data || {}; // Get denormalized values from backend
    
    const dynamicColor = getProbabilityColor ? getProbabilityColor(prob, 1) : '#3b82f6';

    // Data for the Radar Chart - Scaled to fit 0-100 range visually
    const radarData = [
        { subject: 'NDVI', value: (sensors.ndvi16 ?? 0) * 100 },
        { subject: 'Temp', value: (sensors.temp16 ?? 0) * 2 },
        { subject: 'Precip', value: (sensors.precip16 ?? 0) * 15 },
        { subject: 'Humidity', value: (sensors.rh16 ?? 0) * 100 },
        { subject: 'VPD', value: (sensors.vpd16 ?? 0) * 30 },
        { subject: 'Slope', value: (sensors.slope ?? 0) * 1.4 }
    ];

    const bgColor = isDark ? 'bg-slate-900' : 'bg-white';
    const textColor = isDark ? 'text-white' : 'text-slate-900';
    const borderColor = isDark ? 'border-slate-700' : 'border-slate-200';
    const cardBg = isDark ? 'bg-slate-800' : 'bg-slate-50';

    const getRiskLevel = (p) => {
        if (p < 0.3) return 'LOW RISK';
        if (p < 0.5) return 'MODERATE';
        if (p < 0.7) return 'HIGH RISK';
        return 'CRITICAL';
    };

    return (
        <div className={`fixed right-0 top-0 h-full w-full md:w-96 ${bgColor} ${textColor} shadow-2xl z-[1000] overflow-y-auto border-l ${borderColor} animate-in slide-in-from-right duration-300`}>
            <div className={`sticky top-0 ${bgColor} border-b ${borderColor} p-4 flex justify-between items-center z-10`}>
                <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: dynamicColor }}></div>
                    <h2 className="text-lg font-semibold">Patch Analysis</h2>
                </div>
                <button onClick={onClose} className="p-2 hover:bg-slate-700/20 rounded-full">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>

            <div className="p-4 space-y-6">
                {/* Location Metadata */}
                <div className={`${cardBg} rounded-lg p-4`}>
                    <h3 className="text-xs font-bold text-slate-400 mb-3 tracking-widest uppercase">Location Metadata</h3>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                        <div className="flex flex-col">
                            <span className="text-slate-500 text-[10px]">Patch ID</span>
                            <span className="font-mono text-xs">{patch.id}</span>
                        </div>
                        <div className="col-span-2">
                            <span className="text-slate-500 text-[10px]">Coordinates</span>
                            <div className="font-mono text-xs text-blue-400">{patch.lat?.toFixed(4)}°N, {patch.lng?.toFixed(4)}°E</div>
                        </div>
                    </div>
                </div>

                {/* Fire Probability Bar */}
                <div className={`${cardBg} rounded-lg p-4 border-l-4`} style={{ borderLeftColor: dynamicColor }}>
                    <h3 className="text-xs font-bold text-slate-400 mb-3 tracking-widest uppercase">Fire Probability</h3>
                    <div className="relative h-3 bg-slate-700/50 rounded-full overflow-hidden">
                        <div className="absolute top-0 h-full transition-all duration-1000" style={{ width: `${prob * 100}%`, backgroundColor: dynamicColor }} />
                    </div>
                    <div className="flex justify-between mt-3 items-end">
                        <span className="text-3xl font-bold" style={{ color: dynamicColor }}>{(prob * 100).toFixed(1)}%</span>
                        <div className="px-3 py-1 rounded-md text-[10px] font-black tracking-tighter uppercase" style={{ backgroundColor: `${dynamicColor}20`, color: dynamicColor, border: `1px solid ${dynamicColor}40` }}>
                            {getRiskLevel(prob)}
                        </div>
                    </div>
                </div>

                {/* Radar Chart */}
                <div className={`${cardBg} rounded-lg p-4`}>
                    <h3 className="text-xs font-bold text-slate-400 mb-3 tracking-widest uppercase">Risk Vectors</h3>
                    <div className="h-[220px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <RadarChart data={radarData}>
                                <PolarGrid stroke={isDark ? '#334155' : '#cbd5e1'} />
                                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                                <Radar dataKey="value" stroke={dynamicColor} fill={dynamicColor} fillOpacity={0.3} />
                            </RadarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Real Sensor Readings from Backend */}
                <div className={`${cardBg} rounded-lg p-4`}>
                    <h3 className="text-xs font-bold text-slate-400 mb-4 tracking-widest text-center uppercase">Actual Sensor Readings</h3>
                    <div className="grid grid-cols-1 gap-y-3 text-sm">
                        <div className="flex justify-between items-center border-b border-slate-700/30 pb-1">
                            <span className="text-slate-400">Elevation</span>
                            <span className="font-mono">{sensors.elevation?.toLocaleString() ?? '--'} m</span>
                        </div>
                        <div className="flex justify-between items-center border-b border-slate-700/30 pb-1">
                            <span className="text-slate-400">Slope</span>
                            <span className="font-mono">{sensors.slope?.toFixed(1) ?? '--'}°</span>
                        </div>
                        <div className="flex justify-between items-center border-b border-slate-700/30 pb-1">
                            <span className="text-slate-400">NDVI (Vegetation)</span>
                            <span className="font-mono text-green-400">{sensors.ndvi16?.toFixed(3) ?? 'N/A'}</span>
                        </div>
                        <div className="flex justify-between items-center border-b border-slate-700/30 pb-1">
                            <span className="text-slate-400">Temperature</span>
                            <span className="font-mono text-orange-400">{sensors.temp16?.toFixed(1) ?? '--'}°C</span>
                        </div>
                        <div className="flex justify-between items-center border-b border-slate-700/30 pb-1">
                            <span className="text-slate-400">Humidity</span>
                            <span className="font-mono text-blue-400">{(sensors.rh16 * 100).toFixed(0) ?? '--'}%</span>
                        </div>
                        <div className="flex justify-between items-center border-b border-slate-700/30 pb-1">
                            <span className="text-slate-400">Precipitation</span>
                            <span className="font-mono text-cyan-400">{sensors.precip16?.toFixed(2) ?? '--'} mm</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-slate-400">VPD (Air Dryness)</span>
                            <span className="font-mono">{sensors.vpd16?.toFixed(2) ?? 'N/A'} kPa</span>
                        </div>
                    </div>
                </div>
                
                <div className="text-[9px] text-center text-slate-500 pb-4">
                    Model: {modelVersion} • Frequency: 24h
                </div>
            </div>
        </div>
    );
};

export default SidePanel;