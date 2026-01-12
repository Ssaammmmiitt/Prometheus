const Legend = ({ threshold, showProbability ,useTheme}) => {
const { isDark } = useTheme();
const bgColor = isDark ? 'bg-slate-900/95' : 'bg-white/95';
const textColor = isDark ? 'text-white' : 'text-slate-900';

return (
    <div className={`absolute bottom-8 left-4 ${bgColor} ${textColor} backdrop-blur-sm rounded-xl shadow-xl p-4 z-900`}>
        <h4 className="text-sm font-medium mb-2">Fire Probability</h4>
        <div className="flex items-center gap-2">
            <div className="w-32 h-3 rounded bg-linear-to-r from-green-500  via-orange-500 to-red-500 relative">
                <div 
                    className="absolute top-0 h-full w-0.5 bg-white shadow"
                    style={{ left: `${threshold * 100}%` }}
                />
            </div>
        </div>
        <div className="flex justify-between text-xs text-slate-400 mt-1">
            <span>0%</span>
            <span>100%</span>
        </div>
        <p className="text-xs text-slate-400 mt-2">Threshold: {(threshold * 100).toFixed(0)}%</p>
    </div>
);
};


export default Legend;