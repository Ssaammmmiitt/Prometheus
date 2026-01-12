import React, { useState, useEffect, useCallback, createContext, useContext } from 'react';
import { MapContainer, TileLayer, useMap, Popup } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';



import StatsBar from './Components/StatsBar';
import Legend from './Components/Legend';
import PatchGridLayer from './Components/PatchGridLayer';
import SidePanel from './Components/SidePanel';
import ZoomControls from './Components/ZoomControls';
import ControlPanel from './Components/ControlPanel';
import HistoricalFiresLayer from './Components/HistoricalFiresLayer';

// Theme Context
const ThemeContext = createContext({ isDark: true, toggleTheme: () => {} });

const useTheme = () => useContext(ThemeContext);

// Nepal bounds and configuration
const NEPAL_BOUNDS = {
north: 30.45,
south: 26.35,
east: 88.20,
west: 80.05
};

const NEPAL_CENTER = [28.3949, 84.1240];
const PATCH_SIZE_KM = 32;
const PATCH_ROWS = Math.ceil((NEPAL_BOUNDS.north - NEPAL_BOUNDS.south) * 111 / PATCH_SIZE_KM);
const PATCH_COLS = Math.ceil((NEPAL_BOUNDS.east - NEPAL_BOUNDS.west) * 85 / PATCH_SIZE_KM);

// Mock data generator with realistic coordinates
const generatePatches = () => {
const patches = [];
const latStep = (NEPAL_BOUNDS.north - NEPAL_BOUNDS.south) / PATCH_ROWS;
const lngStep = (NEPAL_BOUNDS.east - NEPAL_BOUNDS.west) / PATCH_COLS;

for (let row = 0; row < PATCH_ROWS; row++) {
    for (let col = 0; col < PATCH_COLS; col++) {
        const lat = NEPAL_BOUNDS.south + (row + 0.5) * latStep;
        const lng = NEPAL_BOUNDS.west + (col + 0.5) * lngStep;
        
        if (isInsideNepal(lat, lng)) {
            patches.push({
                id: `patch_${row}_${col}`,
                row,
                col,
                lat,
                lng,
                bounds: [
                    [NEPAL_BOUNDS.south + row * latStep, NEPAL_BOUNDS.west + col * lngStep],
                    [NEPAL_BOUNDS.south + (row + 1) * latStep, NEPAL_BOUNDS.west + (col + 1) * lngStep]
                ],
                probability: Math.random() * 0.8 + 0.1,
                elevation: Math.floor(Math.random() * 6000 + 500),
                slope: Math.floor(Math.random() * 45),
                ndvi: Math.random() * 0.6 + 0.2,
                temperature: Math.floor(Math.random() * 25 + 5),
                precipitation: Math.floor(Math.random() * 200),
                humidity: Math.floor(Math.random() * 60 + 30),
                vpd: Math.random() * 2 + 0.5,
                district: getDistrictName(lat, lng),
                landCover: getLandCover()
            });
        }
    }
}
return patches;
};

const isInsideNepal = (lat, lng) => {
// Simplified Nepal boundary check
if (lat < 26.35 || lat > 30.45 || lng < 80.05 || lng > 88.20) return false;

// Rough shape approximation
const normalizedLat = (lat - 26.35) / (30.45 - 26.35);
const normalizedLng = (lng - 80.05) / (88.20 - 80.05);

if (normalizedLng < 0.1 && normalizedLat > 0.7) return false;
if (normalizedLng > 0.9 && normalizedLat > 0.6) return false;
if (normalizedLng < 0.2 && normalizedLat < 0.3) return false;

return true;
};

const getDistrictName = (lat, lng) => {
const districts = ['Kathmandu', 'Pokhara', 'Chitwan', 'Bhaktapur', 'Lalitpur', 'Makwanpur', 'Kaski', 'Rupandehi'];
return districts[Math.floor(Math.random() * districts.length)];
};

const getLandCover = () => {
const covers = ['Forest', 'Grassland', 'Shrubland', 'Agricultural', 'Mixed'];
return covers[Math.floor(Math.random() * covers.length)];
};

const generateHistoricalFires = () => {
const fires = [];
for (let year = 2018; year <= 2024; year++) {
    const count = Math.floor(Math.random() * 50 + 20);
    for (let i = 0; i < count; i++) {
        const lat = NEPAL_BOUNDS.south + Math.random() * (NEPAL_BOUNDS.north - NEPAL_BOUNDS.south);
        const lng = NEPAL_BOUNDS.west + Math.random() * (NEPAL_BOUNDS.east - NEPAL_BOUNDS.west);
        if (isInsideNepal(lat, lng)) {
            fires.push({
                id: `fire_${year}_${i}`,
                year,
                lat,
                lng,
                frp: Math.random() * 100 + 10,
                date: `${year}-${String(Math.floor(Math.random() * 5) + 1).padStart(2, '0')}-${String(Math.floor(Math.random() * 28) + 1).padStart(2, '0')}`
            });
        }
    }
}
return fires;
};

// Probability color scale
const getProbabilityColor = (prob, opacity = 0.3) => {
if (prob < 0.3) return `rgba(34, 197, 94, ${opacity})`;
if (prob < 0.5) return `rgba(234, 179, 8, ${opacity})`;
if (prob < 0.7) return `rgba(249, 115, 22, ${opacity})`;
return `rgba(239, 68, 68, ${opacity})`;
};

// Map Controller Component
const MapController = ({ selectedPatch, flyToNepal }) => {
const map = useMap();

useEffect(() => {
    if (flyToNepal) {
        map.flyTo(NEPAL_CENTER, 7, { duration: 2 });
    }
}, [flyToNepal, map]);

useEffect(() => {
    if (selectedPatch) {
        map.flyTo([selectedPatch.lat, selectedPatch.lng], 10, { duration: 1 });
    }
}, [selectedPatch, map]);

return null;
};

// Patch Grid Layer

// Historical Fires Layer

// Side Panel Component


// Control Panel Component


// Legend Component


// Zoom Controls Component


// Stats Bar Component

// Main Dashboard Component
export default function WildfireDashboard() {
const [isDark, setIsDark] = useState(() => {
    if (typeof window !== 'undefined') {
        return localStorage.getItem('theme') !== 'light';
    }
    return true;
});

const toggleTheme = useCallback(() => {
    setIsDark(prev => {
        const newValue = !prev;
        localStorage.setItem('theme', newValue ? 'dark' : 'light');
        return newValue;
    });
}, []);

const [patches] = useState(() => generatePatches());
const [historicalFires] = useState(() => generateHistoricalFires());

const [selectedPatch, setSelectedPatch] = useState(null);
const [hoveredPatch, setHoveredPatch] = useState(null);
const [flyToNepal, setFlyToNepal] = useState(true);

const [predictionDate, setPredictionDate] = useState(new Date().toISOString().split('T')[0]);
const [historicalYear, setHistoricalYear] = useState(2024);
const [threshold, setThreshold] = useState(0.5);
const [modelVersion, setModelVersion] = useState('v3.2.1');
const [showProbability, setShowProbability] = useState(true);
const [showHistorical, setShowHistorical] = useState(false);
const [probabilityOpacity, setProbabilityOpacity] = useState(0.5);

const handlePatchClick = useCallback((patch) => {
    setSelectedPatch(patch);
}, []);

const handleResetView = useCallback(() => {
    setFlyToNepal(false);
    setTimeout(() => setFlyToNepal(true), 100);
}, []);

const bgColor = isDark ? 'bg-slate-950' : 'bg-slate-100';
const textColor = isDark ? 'text-white' : 'text-slate-900';

return (
    <ThemeContext.Provider value={{ isDark, toggleTheme }}>
        <div className={`h-screen w-screen overflow-hidden ${bgColor} ${textColor}`}>
            {/* Header */}
            <header className={`absolute top-0 left-0 right-0 h-14 ${isDark ? 'bg-slate-900/90' : 'bg-white/90'} backdrop-blur-sm z-950 flex items-center px-6 border-b ${isDark ? 'border-slate-800' : 'border-slate-200'}`}>
                <div className="flex items-center gap-3">
                    <span className="text-2xl">🔥</span>
                    <div>
                        <h1 className="font-bold text-lg">Nepal Wildfire Risk Dashboard</h1>
                        <p className="text-xs text-slate-400">Geospatial Prediction & Exploration</p>
                    </div>
                </div>
            </header>

            {/* Map Container */}
            <div className="absolute inset-0 pt-14">
                <MapContainer
                    center={[20, 0]}
                    zoom={1}
                    className="h-full w-full"
                    zoomControl={false}
                >
                    <TileLayer
                        url={isDark 
                            ? "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                            : "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
                        }
                        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                    />
                    
                    <MapController selectedPatch={selectedPatch} flyToNepal={flyToNepal} />
                    
                    <PatchGridLayer
                        getProbabilityColor={getProbabilityColor}
                        useTheme={useTheme}
                        patches={patches}
                        hoveredPatch={hoveredPatch}
                        setHoveredPatch={setHoveredPatch}
                        onPatchClick={handlePatchClick}
                        showProbability={showProbability}
                        threshold={threshold}
                    />

                    {showHistorical && (
                        <HistoricalFiresLayer fires={historicalFires} selectedYear={historicalYear} />
                    )}

                    <ZoomControls useTheme={useTheme} onResetView={handleResetView} useMap={useMap}/>
                </MapContainer>

                {/* Controls */}
                <ControlPanel
                    useTheme={useTheme}
                    predictionDate={predictionDate}
                    setPredictionDate={setPredictionDate}
                    historicalYear={historicalYear}
                    setHistoricalYear={setHistoricalYear}
                    threshold={threshold}
                    setThreshold={setThreshold}
                    modelVersion={modelVersion}
                    setModelVersion={setModelVersion}
                    showProbability={showProbability}
                    setShowProbability={setShowProbability}
                    showHistorical={showHistorical}
                    setShowHistorical={setShowHistorical}
                    probabilityOpacity={probabilityOpacity}
                    setProbabilityOpacity={setProbabilityOpacity}
                />

                {/* Legend */}
                <Legend useTheme={useTheme} threshold={threshold} showProbability={showProbability} />

                {/* Stats Bar */}
                <StatsBar useTheme={useTheme} patches={patches} threshold={threshold} />

                {/* Hover Tooltip */}
                {hoveredPatch && !selectedPatch && (
                    <div className={`absolute bottom-24 left-1/2 -translate-x-1/2 ${isDark ? 'bg-slate-900' : 'bg-white'} ${textColor} px-4 py-2 rounded-lg shadow-xl z-900 text-sm`}>
                        <p><strong>{hoveredPatch.id}</strong> | {hoveredPatch.lat.toFixed(4)}°N, {hoveredPatch.lng.toFixed(4)}°E</p>
                        <p>Probability: <span className="font-bold">{(hoveredPatch.probability * 100).toFixed(1)}%</span></p>
                    </div>
                )}
            </div>

            {/* Side Panel */}
            {selectedPatch && (
                <SidePanel
                    useTheme={useTheme}
                    patch={selectedPatch}
                    onClose={() => setSelectedPatch(null)}
                    threshold={threshold}
                    modelVersion={modelVersion}
                />
            )}
        </div>
    </ThemeContext.Provider>
);
}