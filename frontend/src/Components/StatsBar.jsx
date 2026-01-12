const StatsBar = ({ patches, threshold,useTheme }) => {
const { isDark } = useTheme();
const bgColor = isDark ? 'bg-slate-900/95' : 'bg-white/95';
const textColor = isDark ? 'text-white' : 'text-slate-900';

const highRisk = patches.filter(p => p.probability >= threshold).length;
const moderateRisk = patches.filter(p => p.probability >= 0.3 && p.probability < threshold).length;
const avgProbability = patches.reduce((sum, p) => sum + p.probability, 0) / patches.length;

return (
    <div className={`absolute bottom-8 right-4 ${bgColor} ${textColor} backdrop-blur-sm rounded-xl shadow-xl p-4 z-900 hidden md:block`}>
        <div className="grid grid-cols-2 gap-4 text-center">
            <div>
                <p className="text-2xl font-bold">{patches.length}</p>
                <p className="text-xs text-slate-400">Total Patches</p>
            </div>
            <div>
                <p className="text-2xl font-bold text-red-400">{highRisk}</p>
                <p className="text-xs text-slate-400">High Risk</p>
            </div>
            <div>
                <p className="text-2xl font-bold text-yellow-400">{moderateRisk}</p>
                <p className="text-xs text-slate-400">Moderate Risk</p>
            </div>
            <div>
                <p className="text-2xl font-bold text-orange-400">{(avgProbability * 100).toFixed(1)}%</p>
                <p className="text-xs text-slate-400">Avg Probability</p>
            </div>
        </div>
    </div>
);
};


export default StatsBar