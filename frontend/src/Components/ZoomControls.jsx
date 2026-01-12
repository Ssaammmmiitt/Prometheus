const ZoomControls = ({ onResetView,useMap ,useTheme}) => {
const { isDark, toggleTheme } = useTheme();
const map = useMap();
const bgColor = isDark ? 'bg-slate-900' : 'bg-white';
const textColor = isDark ? 'text-white' : 'text-slate-900';
const hoverBg = isDark ? 'hover:bg-slate-700' : 'hover:bg-slate-100';

return (
    <div className={`absolute top-4 right-4 ${bgColor} ${textColor} rounded-xl shadow-xl z-900 flex flex-col overflow-hidden`}>
        <button
            onClick={() => map.zoomIn()}
            className={`p-3 ${hoverBg} transition-colors border-b ${isDark ? 'border-slate-700' : 'border-slate-200'}`}
            title="Zoom In"
        >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
        </button>
        <button
            onClick={() => map.zoomOut()}
            className={`p-3 ${hoverBg} transition-colors border-b ${isDark ? 'border-slate-700' : 'border-slate-200'}`}
            title="Zoom Out"
        >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
            </svg>
        </button>
        <button
            onClick={onResetView}
            className={`p-3 ${hoverBg} transition-colors border-b ${isDark ? 'border-slate-700' : 'border-slate-200'}`}
            title="Reset View to Nepal"
        >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
            </svg>
        </button>
        <button
            onClick={toggleTheme}
            className={`p-3 ${hoverBg} transition-colors`}
            title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
            {isDark ? (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
            ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
            )}
        </button>
    </div>
);
};

export default ZoomControls