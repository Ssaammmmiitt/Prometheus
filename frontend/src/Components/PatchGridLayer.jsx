import { GeoJSON } from "react-leaflet";

const PatchGridLayer = ({useTheme,getProbabilityColor, patches, hoveredPatch, setHoveredPatch, onPatchClick, showProbability, threshold }) => {
const { isDark } = useTheme();

return (
    <>
        {patches.map(patch => {
            const isHovered = hoveredPatch?.id === patch.id;
            const isHighRisk = patch.probability >= threshold;
            
            return (
                <GeoJSON
                    key={patch.id}
                    data={{
                        type: 'Feature',
                        geometry: {
                            type: 'Polygon',
                            coordinates: [[
                                [patch.bounds[0][1], patch.bounds[0][0]],
                                [patch.bounds[1][1], patch.bounds[0][0]],
                                [patch.bounds[1][1], patch.bounds[1][0]],
                                [patch.bounds[0][1], patch.bounds[1][0]],
                                [patch.bounds[0][1], patch.bounds[0][0]]
                            ]]
                        },
                        properties: patch
                    }}
                    style={{
                        fillColor: showProbability ? getProbabilityColor(patch.probability, isHovered ? 0.5 : 0.25) : (isDark ? 'rgba(100, 116, 139, 0.15)' : 'rgba(148, 163, 184, 0.2)'),
                        fillOpacity: 1,
                        color: isHovered ? (isDark ? '#60a5fa' : '#3b82f6') : (isDark ? 'rgba(148, 163, 184, 0.3)' : 'rgba(100, 116, 139, 0.3)'),
                        weight: isHovered ? 2 : 1
                    }}
                    eventHandlers={{
                        mouseover: () => setHoveredPatch(patch),
                        mouseout: () => setHoveredPatch(null),
                        click: () => onPatchClick(patch)
                    }}
                />
            );
        })}
    </>
);
};


export default PatchGridLayer;