const ControlPanel = ({ useTheme,
predictionDate, setPredictionDate,
historicalYear, setHistoricalYear,
threshold, setThreshold,
modelVersion, setModelVersion,
showProbability, setShowProbability,
showHistorical, setShowHistorical,
probabilityOpacity, setProbabilityOpacity
}) => {
const { isDark } = useTheme();
const bgColor = isDark ? 'bg-slate-900/95' : 'bg-white/95';
const textColor = isDark ? 'text-white' : 'text-slate-900';
const borderColor = isDark ? 'border-slate-700' : 'border-slate-200';

return (
    <div className={`absolute top-4 left-4 ${bgColor} ${textColor} backdrop-blur-sm rounded-xl shadow-xl p-4 z-900 w-72 max-h-[calc(100vh-120px)] overflow-y-auto`}>
        <h3 className="font-semibold mb-4 text-lg">Dashboard Controls</h3>
        
        <div className="space-y-4">
            {/* Prediction Date */}
            <div>
                <label className="text-sm text-slate-400 block mb-1">Prediction Window (16-day)</label>
                <input
                    type="date"
                    value={predictionDate}
                    onChange={(e) => setPredictionDate(e.target.value)}
                    className={`w-full px-3 py-2 rounded-lg border ${borderColor} ${isDark ? 'bg-slate-800' : 'bg-slate-50'} text-sm`}
                />
            </div>

            {/* Model Version */}
            <div>
                <label className="text-sm text-slate-400 block mb-1">Model Version</label>
                <select
                    value={modelVersion}
                    onChange={(e) => setModelVersion(e.target.value)}
                    className={`w-full px-3 py-2 rounded-lg border ${borderColor} ${isDark ? 'bg-slate-800' : 'bg-slate-50'} text-sm`}
                >
                    <option value="v3.2.1">v3.2.1 (Latest)</option>
                    <option value="v3.1.0">v3.1.0</option>
                    <option value="v3.0.0">v3.0.0</option>
                </select>
            </div>

            {/* Threshold Slider */}
            <div>
                <label className="text-sm text-slate-400 block mb-1">
                    Risk Threshold: {(threshold * 100).toFixed(0)}%
                </label>
                <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="w-full accent-orange-500"
                />
            </div>

            {/* Layer Toggles */}
            <div className={`border-t ${borderColor} pt-4`}>
                <h4 className="text-sm font-medium mb-3">Layers</h4>
                
                <div className="space-y-3">
                    <label className="flex items-center gap-2 cursor-pointer">
                        <input
                            type="checkbox"
                            checked={showProbability}
                            onChange={(e) => setShowProbability(e.target.checked)}
                            className="accent-orange-500"
                        />
                        <span className="text-sm">Probability Overlay</span>
                    </label>
                    
                    {showProbability && (
                        <div className="ml-6">
                            <label className="text-xs text-slate-400">Opacity: {Math.round(probabilityOpacity * 100)}%</label>
                            <input
                                type="range"
                                min="0.1"
                                max="1"
                                step="0.1"
                                value={probabilityOpacity}
                                onChange={(e) => setProbabilityOpacity(parseFloat(e.target.value))}
                                className="w-full accent-orange-500"
                            />
                        </div>
                    )}

                    <label className="flex items-center gap-2 cursor-pointer">
                        <input
                            type="checkbox"
                            checked={showHistorical}
                            onChange={(e) => setShowHistorical(e.target.checked)}
                            className="accent-red-500"
                        />
                        <span className="text-sm">Historical Fires</span>
                    </label>

                    {showHistorical && (
                        <div className="ml-6">
                            <select
                                value={historicalYear}
                                onChange={(e) => setHistoricalYear(parseInt(e.target.value))}
                                className={`w-full px-2 py-1 rounded border ${borderColor} ${isDark ? 'bg-slate-800' : 'bg-slate-50'} text-sm`}
                            >
                                {[2024, 2023, 2022, 2021, 2020, 2019, 2018].map(year => (
                                    <option key={year} value={year}>{year}</option>
                                ))}
                            </select>
                        </div>
                    )}
                </div>
            </div>
        </div>
    </div>
);
};


export default ControlPanel;