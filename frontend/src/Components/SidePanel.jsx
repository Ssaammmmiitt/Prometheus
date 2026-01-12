import {
    AreaChart, Area, BarChart, Bar, RadarChart, Radar, PolarGrid,
    PolarAngleAxis, PolarRadiusAxis, XAxis, YAxis, Tooltip, ResponsiveContainer
} from 'recharts';


const SidePanel = ({ patch, onClose,useTheme, threshold, modelVersion }) => {
const { isDark } = useTheme();
const isHighRisk = patch.probability >= threshold;

const trendData = Array.from({ length: 30 }, (_, i) => ({
    day: i + 1,
    probability: Math.max(0, Math.min(1, patch.probability + (Math.random() - 0.5) * 0.3))
}));

const seasonalData = [
    { month: 'Jan', risk: 0.7 },
    { month: 'Feb', risk: 0.8 },
    { month: 'Mar', risk: 0.9 },
    { month: 'Apr', risk: 0.85 },
    { month: 'May', risk: 0.6 },
    { month: 'Jun', risk: 0.2 }
];

const radarData = [
    { subject: 'NDVI', value: patch.ndvi * 100, fullMark: 100 },
    { subject: 'Temp', value: patch.temperature * 2, fullMark: 100 },
    { subject: 'Precip', value: patch.precipitation / 2, fullMark: 100 },
    { subject: 'Humidity', value: patch.humidity, fullMark: 100 },
    { subject: 'VPD', value: patch.vpd * 40, fullMark: 100 },
    { subject: 'Slope', value: patch.slope * 2, fullMark: 100 }
];

const bgColor = isDark ? 'bg-slate-900' : 'bg-white';
const textColor = isDark ? 'text-white' : 'text-slate-900';
const borderColor = isDark ? 'border-slate-700' : 'border-slate-200';
const cardBg = isDark ? 'bg-slate-800' : 'bg-slate-50';

return (
    <div className={`fixed right-0 top-0 h-full w-full md:w-96 ${bgColor} ${textColor} shadow-2xl z-1000 overflow-y-auto`}>
        <div className={`sticky top-0 ${bgColor} border-b ${borderColor} p-4 flex justify-between items-center`}>
            <h2 className="text-lg font-semibold">Patch Details</h2>
            <button onClick={onClose} className="p-2 hover:bg-slate-700 rounded-full transition-colors">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>

        <div className="p-4 space-y-6">
            {/* Metadata */}
            <div className={`${cardBg} rounded-lg p-4`}>
                <h3 className="text-sm font-medium text-slate-400 mb-3">PATCH METADATA</h3>
                <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                        <span className="text-slate-400">ID:</span>
                        <span className="ml-2 font-mono">{patch.id}</span>
                    </div>
                    <div>
                        <span className="text-slate-400">Grid:</span>
                        <span className="ml-2">({patch.row}, {patch.col})</span>
                    </div>
                    <div className="col-span-2">
                        <span className="text-slate-400">Location:</span>
                        <span className="ml-2 font-mono">{patch.lat.toFixed(4)}°N, {patch.lng.toFixed(4)}°E</span>
                    </div>
                    <div>
                        <span className="text-slate-400">District:</span>
                        <span className="ml-2">{patch.district}</span>
                    </div>
                    <div>
                        <span className="text-slate-400">Land Cover:</span>
                        <span className="ml-2">{patch.landCover}</span>
                    </div>
                </div>
            </div>

            {/* Fire Probability Gauge */}
            <div className={`${cardBg} rounded-lg p-4`}>
                <h3 className="text-sm font-medium text-slate-400 mb-3">FIRE PROBABILITY</h3>
                <div className="relative h-4 bg-linear-to-r from-green-500  via-orange-500 to-red-500 rounded-full overflow-hidden">
                    <div 
                        className="absolute top-0 h-full w-1 bg-white shadow-lg transition-all duration-500"
                        style={{ left: `${patch.probability * 100}%` }}
                    />
                </div>
                <div className="flex justify-between mt-2 text-sm">
                    <span className="text-green-400">Low</span>
                    <span className="font-bold text-2xl">{(patch.probability * 100).toFixed(1)}%</span>
                    <span className="text-red-400">High</span>
                </div>
                <div className={`mt-3 px-3 py-2 rounded-lg text-center font-medium ${isHighRisk ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                    {isHighRisk ? '⚠️ HIGH RISK' : '✓ LOW RISK'}
                </div>
                <p className="text-xs text-slate-400 mt-2 text-center">Model: {modelVersion}</p>
            </div>

            {/* Trend Chart */}
            <div className={`${cardBg} rounded-lg p-4`}>
                <h3 className="text-sm font-medium text-slate-400 mb-3">30-DAY TREND</h3>
                <ResponsiveContainer width="100%" height={120}>
                    <AreaChart data={trendData}>
                        <defs>
                            <linearGradient id="probGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.8}/>
                                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <XAxis dataKey="day" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                        <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                        <Tooltip contentStyle={{ backgroundColor: isDark ? '#1e293b' : '#fff', border: 'none' }} />
                        <Area type="monotone" dataKey="probability" stroke="#f59e0b" fill="url(#probGradient)" />
                    </AreaChart>
                </ResponsiveContainer>
            </div>

            {/* Seasonal Pattern */}
            <div className={`${cardBg} rounded-lg p-4`}>
                <h3 className="text-sm font-medium text-slate-400 mb-3">SEASONAL PATTERN</h3>
                <ResponsiveContainer width="100%" height={120}>
                    <BarChart data={seasonalData}>
                        <XAxis dataKey="month" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                        <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                        <Tooltip contentStyle={{ backgroundColor: isDark ? '#1e293b' : '#fff', border: 'none' }} />
                        <Bar dataKey="risk" fill="#ef4444" radius={[4, 4, 0, 0]} />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Environmental Radar */}
            <div className={`${cardBg} rounded-lg p-4`}>
                <h3 className="text-sm font-medium text-slate-400 mb-3">ENVIRONMENTAL INPUTS</h3>
                <ResponsiveContainer width="100%" height={200}>
                    <RadarChart data={radarData}>
                        <PolarGrid stroke={isDark ? '#475569' : '#cbd5e1'} />
                        <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 8, fill: '#94a3b8' }} />
                        <Radar name="Values" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.5} />
                    </RadarChart>
                </ResponsiveContainer>
            </div>

            {/* Environmental Details */}
            <div className={`${cardBg} rounded-lg p-4`}>
                <h3 className="text-sm font-medium text-slate-400 mb-3">DETAILED VALUES</h3>
                <div className="space-y-2 text-sm">
                    <div className="flex justify-between"><span className="text-slate-400">Elevation</span><span>{patch.elevation} m</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Slope</span><span>{patch.slope}°</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">NDVI</span><span>{patch.ndvi.toFixed(3)}</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Temperature</span><span>{patch.temperature}°C</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Precipitation</span><span>{patch.precipitation} mm</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Humidity</span><span>{patch.humidity}%</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">VPD</span><span>{patch.vpd.toFixed(2)} kPa</span></div>
                </div>
            </div>
        </div>
    </div>
);
};

export default SidePanel;