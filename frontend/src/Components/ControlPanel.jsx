"use client";
import React from 'react';

const ControlPanel = ({ 
    useTheme,
    predictionDate, setPredictionDate,
    historicalYear, setHistoricalYear,
    threshold, setThreshold,
    showProbability, setShowProbability,
    showHistorical, setShowHistorical,
    probabilityOpacity, setProbabilityOpacity
}) => {
    const { isDark } = useTheme();
    
    // UI Constants
    const bgColor = isDark ? 'bg-slate-900/95' : 'bg-white/95';
    const textColor = isDark ? 'text-white' : 'text-slate-900';
    const borderColor = isDark ? 'border-slate-700' : 'border-slate-200';
    const inputBg = isDark ? 'bg-slate-800' : 'bg-slate-50';

    // --- Boundary Snap Logic ---
    const handleDateChange = (e) => {
        const selectedValue = e.target.value;
        if (!selectedValue) return;

        const date = new Date(selectedValue);
        const year = date.getFullYear();
        const month = date.getMonth(); // 0 = Jan, 2 = Mar, 4 = May, 11 = Dec

        let snappedDate = selectedValue;

        // If user selects Jan (0) or Feb (1) -> Snap to March 1st
        if (month < 2) {
            snappedDate = `${year}-03-01`;
        } 
        // If user selects June (5) through Dec (11) -> Snap to May 31st
        else if (month > 4) {
            snappedDate = `${year}-05-31`;
        }

        setPredictionDate(snappedDate);
    };

    return (
        <div className={`absolute top-4 left-4 ${bgColor} ${textColor} backdrop-blur-sm rounded-xl shadow-xl p-4 z-[1000] w-72 max-h-[calc(100vh-120px)] overflow-y-auto border ${borderColor}`}>
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-lg">Analysis Controls</h3>
                <span className="text-[10px] bg-orange-500/20 text-orange-500 px-2 py-0.5 rounded-full font-bold uppercase tracking-tighter">Live Data</span>
            </div>
            
            <div className="space-y-4">
                {/* Date Selection with Boundary Snapping */}
                <div>
                    <label className="text-sm text-slate-400 block mb-1 font-medium flex justify-between">
                        <span>Prediction Window</span>
                    </label>
                    <input
                        type="date"
                        value={predictionDate}
                        min="2018-01-01"
                        max="2025-12-31"
                        onChange={handleDateChange}
                        className={`w-full px-3 py-2 rounded-lg border ${borderColor} ${inputBg} text-sm focus:ring-2 focus:ring-orange-500 outline-none`}
                        style={{ colorScheme: isDark ? 'dark' : 'light' }}
                    />
                    
                </div>

                {/* Risk Sensitivity Slider */}
                {/* <div>
                    <label className="text-sm text-slate-400 block mb-1 flex justify-between">
                        <span>Risk Sensitivity</span>
                        <span className="font-bold text-orange-500">{(threshold * 100).toFixed(0)}%</span>
                    </label>
                    <input
                        type="range"
                        min="0.1"
                        max="0.9"
                        step="0.05"
                        value={threshold}
                        onChange={(e) => setThreshold(parseFloat(e.target.value))}
                        className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500"
                    />
                </div> */}

                <div className={`border-t ${borderColor} pt-4`}>
                    <h4 className="text-sm font-medium mb-3">Map Layers</h4>
                    <div className="space-y-4">
                        <div className="space-y-2">
                            <label className="flex items-center justify-between cursor-pointer group">
                                <span className="text-sm group-hover:text-orange-500 transition-colors">Risk Heatmap</span>
                                <input
                                    type="checkbox"
                                    checked={showProbability}
                                    onChange={(e) => setShowProbability(e.target.checked)}
                                    className="accent-orange-500 h-4 w-4 cursor-pointer"
                                />
                            </label>
                            {/* {showProbability && (
                                <div className="ml-2 pl-4 border-l-2 border-slate-700 space-y-1">
                                    <label className="text-[10px] text-slate-400 uppercase">Layer Opacity</label>
                                    <input
                                        type="range"
                                        min="0.1"
                                        max="1"
                                        step="0.1"
                                        value={probabilityOpacity}
                                        onChange={(e) => setProbabilityOpacity(parseFloat(e.target.value))}
                                        className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500"
                                    />
                                </div>
                            )} */}
                        </div>

                        <div className="space-y-2">
                            <label className="flex items-center justify-between cursor-pointer group">
                                <span className="text-sm group-hover:text-red-500 transition-colors">Historical Archive</span>
                                <input
                                    type="checkbox"
                                    checked={showHistorical}
                                    onChange={(e) => setShowHistorical(e.target.checked)}
                                    className="accent-red-500 h-4 w-4 cursor-pointer"
                                />
                            </label>
                            {showHistorical && (
                                <div className="ml-2 pl-4 border-l-2 border-slate-700">
                                    <select
                                        value={historicalYear}
                                        onChange={(e) => setHistoricalYear(parseInt(e.target.value))}
                                        className={`w-full px-2 py-1.5 rounded border ${borderColor} ${inputBg} text-xs focus:ring-1 focus:ring-red-500 outline-none`}
                                    >
                                        {[2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018].map(year => (
                                            <option key={year} value={year}>{year}</option>
                                        ))}
                                    </select>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ControlPanel;