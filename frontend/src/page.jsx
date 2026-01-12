"use client";
import React, { useState, useEffect, useCallback, createContext, useContext } from 'react';
import { MapContainer, TileLayer, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

// Component Imports
import StatsBar from './Components/StatsBar';
import Legend from './Components/Legend';
import PatchGridLayer from './Components/PatchGridLayer';
import SidePanel from './Components/SidePanel';
import ZoomControls from './Components/ZoomControls';
import ControlPanel from './Components/ControlPanel';
import HistoricalFiresLayer from './Components/HistoricalFiresLayer';

// Data Import
import historicalFiresData from '../../fire_clean_2018_2025.json';

const ThemeContext = createContext({ isDark: true, toggleTheme: () => {} });
const useTheme = () => useContext(ThemeContext);

const NEPAL_BOUNDS = { north: 30.45, south: 26.35, east: 88.20, west: 80.05 };
const NEPAL_CENTER = [28.3949, 84.1240];
const PATCH_SIZE_KM = 32;

// --- Helper Functions ---
const isInsideNepal = (lat, lng) => {
    if (lat < 26.3 || lat > 30.5 || lng < 80.0 || lng > 88.3) return false;
    const northLimit = 30.6 - ((lng - 80.0) * 0.28);
    if (lat > northLimit) return false;
    return true;
};

const getProbabilityColor = (prob, opacity = 0.3) => {
    if (prob < 0.3) return `rgba(34, 197, 94, ${opacity})`;
    if (prob < 0.5) return `rgba(234, 179, 8, ${opacity})`;
    if (prob < 0.7) return `rgba(249, 115, 22, ${opacity})`;
    return `rgba(239, 68, 68, ${opacity})`;
};

const generatePatches = () => {
    const patches = [];
    const rows = Math.ceil((NEPAL_BOUNDS.north - NEPAL_BOUNDS.south) * 111 / PATCH_SIZE_KM);
    const cols = Math.ceil((NEPAL_BOUNDS.east - NEPAL_BOUNDS.west) * 85 / PATCH_SIZE_KM);
    const latStep = (NEPAL_BOUNDS.north - NEPAL_BOUNDS.south) / rows;
    const lngStep = (NEPAL_BOUNDS.east - NEPAL_BOUNDS.west) / cols;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const lat = NEPAL_BOUNDS.south + (r + 0.5) * latStep;
            const lng = NEPAL_BOUNDS.west + (c + 0.5) * lngStep;
            if (isInsideNepal(lat, lng)) {
                patches.push({
                    id: `patch_${r}_${c}`,
                    row: r, col: c, lat, lng,
                    bounds: [[NEPAL_BOUNDS.south + r * latStep, NEPAL_BOUNDS.west + c * lngStep], [NEPAL_BOUNDS.south + (r + 1) * latStep, NEPAL_BOUNDS.west + (c + 1) * lngStep]],
                    probability: 0,
                    sensor_data: null // Initial state
                });
            }
        }
    }
    return patches;
};

const MapController = ({ selectedPatch, flyToNepal }) => {
    const map = useMap();
    useEffect(() => { if (flyToNepal) map.flyTo(NEPAL_CENTER, 7, { duration: 2 }); }, [flyToNepal, map]);
    useEffect(() => { if (selectedPatch) map.flyTo([selectedPatch.lat, selectedPatch.lng], 10, { duration: 1.5 }); }, [selectedPatch, map]);
    return null;
};

export default function WildfireDashboard() {
    const [isDark, setIsDark] = useState(true);
    const [patches, setPatches] = useState(() => generatePatches());
    const [historicalFires] = useState(historicalFiresData);
    
    const [selectedPatch, setSelectedPatch] = useState(null);
    const [loadingPatchId, setLoadingPatchId] = useState(null);
    const [flyToNepal, setFlyToNepal] = useState(true);

    const [predictionDate, setPredictionDate] = useState("2025-04-01");
    const [threshold, setThreshold] = useState(0.5);
    const [showProbability, setShowProbability] = useState(true);
    const [showHistorical, setShowHistorical] = useState(false);
    const [historicalYear, setHistoricalYear] = useState(2024);
    const [probabilityOpacity, setProbabilityOpacity] = useState(0.6);

    const toggleTheme = useCallback(() => setIsDark(prev => !prev), []);

    const handleSetPredictionDate = useCallback((newDate) => {
        if (!newDate) return;
        const date = new Date(newDate);
        const year = date.getFullYear();
        const month = date.getMonth();
        let finalDate = newDate;
        if (month < 2) finalDate = `${year}-03-01`;
        else if (month > 4) finalDate = `${year}-05-31`;
        setPredictionDate(finalDate);
    }, []);

    const handlePatchClick = useCallback(async (clickedPatch) => {
        if (selectedPatch?.id === clickedPatch.id) {
            setSelectedPatch(null);
            return;
        }

        setSelectedPatch(null);
        setLoadingPatchId(clickedPatch.id);

        try {
            const formattedDate = parseInt(predictionDate.replace(/-/g, ''), 10);
            const payload = {
                output_date: formattedDate,
                patch_row: clickedPatch.row,
                patch_col: clickedPatch.col
            };

            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!response.ok) throw new Error("Backend Error");
            const data = await response.json();
            
            if (data.result) {
                const { probability, prediction, sensor_data } = data.result;
                console.log(sensor_data);
                
                // Update patches state with new probability AND sensor data
                setPatches(current => current.map(p => 
                    p.id === clickedPatch.id ? { ...p, probability, predictionResult: prediction, sensor_data } : p
                ));

                const updated = { ...clickedPatch, probability, predictionResult: prediction, sensor_data };
                
                setTimeout(() => {
                    setSelectedPatch(updated);
                    setLoadingPatchId(null);
                }, 5500);
            }
        } catch (err) {
            console.error("❌ Error:", err.message);
            setLoadingPatchId(null);
        }
    }, [selectedPatch, predictionDate]);

    const handleResetView = useCallback(() => {
        setFlyToNepal(false);
        setTimeout(() => setFlyToNepal(true), 100);
    }, []);

    return (
        <ThemeContext.Provider value={{ isDark, toggleTheme }}>
            <div className={`h-screen w-screen overflow-hidden ${isDark ? 'bg-slate-950' : 'bg-slate-100'}`}>
                <header className={`absolute top-0 left-0 right-0 h-14 ${isDark ? 'bg-slate-900/90' : 'bg-white/90'} backdrop-blur-sm z-[1100] flex items-center px-6 border-b ${isDark ? 'border-slate-800' : 'border-slate-200'}`}>
                    <div className="flex items-center gap-3">
                        <span className="text-2xl">🔥</span>
                        <div>
                            <h1 className="font-bold text-lg">Nepal Wildfire Risk Dashboard</h1>
                            <p className="text-[10px] text-slate-400 uppercase tracking-wider">AI Geospatial Engine</p>
                        </div>
                    </div>
                </header>

                <div className="absolute inset-0 pt-14">
                    <MapContainer center={[28, 84]} zoom={7} className="h-full w-full" zoomControl={false}>
                        <TileLayer url={isDark ? "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" : "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"} />
                        <MapController selectedPatch={selectedPatch} flyToNepal={flyToNepal} />
                        <PatchGridLayer
                            patches={patches}
                            onPatchClick={handlePatchClick}
                            getProbabilityColor={getProbabilityColor}
                            showProbability={showProbability}
                            threshold={threshold}
                            selectedPatch={selectedPatch}
                            loadingPatchId={loadingPatchId} 
                        />
                        {showHistorical && <HistoricalFiresLayer fires={historicalFires} selectedYear={historicalYear} />}
                        <ZoomControls useTheme={useTheme} onResetView={handleResetView} useMap={useMap}/>
                    </MapContainer>

                    <ControlPanel
                        useTheme={useTheme}
                        predictionDate={predictionDate} setPredictionDate={handleSetPredictionDate}
                        historicalYear={historicalYear} setHistoricalYear={setHistoricalYear}
                        threshold={threshold} setThreshold={setThreshold}
                        showProbability={showProbability} setShowProbability={setShowProbability}
                        showHistorical={showHistorical} setShowHistorical={setShowHistorical}
                        probabilityOpacity={probabilityOpacity} setProbabilityOpacity={setProbabilityOpacity}
                    />
                    <Legend useTheme={useTheme} threshold={threshold} showProbability={showProbability} />
                    <StatsBar useTheme={useTheme} patches={patches} threshold={threshold} />
                </div>

                {selectedPatch && (
                    <SidePanel
                        useTheme={useTheme}
                        patch={selectedPatch}
                        onClose={() => setSelectedPatch(null)}
                        threshold={threshold}
                        modelVersion="v4.1.0-alpha"
                        getProbabilityColor={getProbabilityColor}
                    />
                )}
            </div>
        </ThemeContext.Provider>
    );
}