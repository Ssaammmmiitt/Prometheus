// import { GeoJSON } from "react-leaflet";

// const PatchGridLayer = ({useTheme,getProbabilityColor, patches, hoveredPatch, setHoveredPatch, onPatchClick, showProbability, threshold }) => {
// const { isDark } = useTheme();

// return (
//     <>
//         {patches.map(patch => {
//             const isHovered = hoveredPatch?.id === patch.id;
//             const isHighRisk = patch.probability >= threshold;
            
//             return (
//                 <GeoJSON
//                     key={patch.id}
//                     data={{
//                         type: 'Feature',
//                         geometry: {
//                             type: 'Polygon',
//                             coordinates: [[
//                                 [patch.bounds[0][1], patch.bounds[0][0]],
//                                 [patch.bounds[1][1], patch.bounds[0][0]],
//                                 [patch.bounds[1][1], patch.bounds[1][0]],
//                                 [patch.bounds[0][1], patch.bounds[1][0]],
//                                 [patch.bounds[0][1], patch.bounds[0][0]]
//                             ]]
//                         },
//                         properties: patch
//                     }}
//                     style={{
//                         fillColor: showProbability ? getProbabilityColor(patch.probability, isHovered ? 0.5 : 0.25) : (isDark ? 'rgba(100, 116, 139, 0.15)' : 'rgba(148, 163, 184, 0.2)'),
//                         fillOpacity: 1,
//                         color: isHovered ? (isDark ? '#60a5fa' : '#3b82f6') : (isDark ? 'rgba(148, 163, 184, 0.3)' : 'rgba(100, 116, 139, 0.3)'),
//                         weight: isHovered ? 2 : 1
//                     }}
//                     eventHandlers={{
//                         mouseover: () => setHoveredPatch(patch),
//                         mouseout: () => setHoveredPatch(null),
//                         click: () => onPatchClick(patch)
//                     }}
//                 />
//             );
//         })}
//     </>
// );
// };


// export default PatchGridLayer;

// Components/PatchGridLayer.jsx
// import { Rectangle } from 'react-leaflet';

// const PatchGridLayer = ({ 
//   patches, 
//   onPatchClick, 
//   getProbabilityColor, 
//   showProbability, 
//   threshold 
// }) => {
//   return (
//     <>
//       {patches.map((patch) => {
//         // Only show if above threshold or if you want to show all
//         if (showProbability && patch.probability < threshold) return null;

//         return (
//           <Rectangle
//             key={patch.id}
//             bounds={patch.bounds}
//             pathOptions={{
//               fillColor: getProbabilityColor(patch.probability),
//               color: 'transparent',
//               fillOpacity: 0.6,
//             }}
//             eventHandlers={{
//               click: () => {
//                 onPatchClick(patch); // This sends the patch data up to the Dashboard
//               },
//               mouseover: (e) => {
//                 e.target.setStyle({ color: '#fff', weight: 2 });
//               },
//               mouseout: (e) => {
//                 e.target.setStyle({ color: 'transparent' });
//               }
//             }}
//           />
//         );
//       })}
//     </>
//   );
// };

// export default PatchGridLayer;

// import { GeoJSON } from "react-leaflet";

// const PatchGridLayer = ({useTheme,getProbabilityColor, patches, hoveredPatch, setHoveredPatch, onPatchClick, showProbability, threshold }) => {
// const { isDark } = useTheme();

// return (
//     <>
//         {patches.map(patch => {
//             const isHovered = hoveredPatch?.id === patch.id;
//             const isHighRisk = patch.probability >= threshold;
            
//             return (
//                 <GeoJSON
//                     key={patch.id}
//                     data={{
//                         type: 'Feature',
//                         geometry: {
//                             type: 'Polygon',
//                             coordinates: [[
//                                 [patch.bounds[0][1], patch.bounds[0][0]],
//                                 [patch.bounds[1][1], patch.bounds[0][0]],
//                                 [patch.bounds[1][1], patch.bounds[1][0]],
//                                 [patch.bounds[0][1], patch.bounds[1][0]],
//                                 [patch.bounds[0][1], patch.bounds[0][0]]
//                             ]]
//                         },
//                         properties: patch
//                     }}
//                     style={{
//                         fillColor: showProbability ? getProbabilityColor(patch.probability, isHovered ? 0.5 : 0.25) : (isDark ? 'rgba(100, 116, 139, 0.15)' : 'rgba(148, 163, 184, 0.2)'),
//                         fillOpacity: 1,
//                         color: isHovered ? (isDark ? '#60a5fa' : '#3b82f6') : (isDark ? 'rgba(148, 163, 184, 0.3)' : 'rgba(100, 116, 139, 0.3)'),
//                         weight: isHovered ? 2 : 1
//                     }}
//                     eventHandlers={{
//                         mouseover: () => setHoveredPatch(patch),
//                         mouseout: () => setHoveredPatch(null),
//                         click: () => onPatchClick(patch)
//                     }}
//                 />
//             );
//         })}
//     </>
// );
// };


// export default PatchGridLayer;

import React from 'react';
import { GeoJSON, Tooltip } from 'react-leaflet';

const PatchGridLayer = ({ 
  patches, 
  onPatchClick, 
  getProbabilityColor, 
  showProbability, 
  selectedPatch, 
  loadingPatchId 
}) => {

  return (
    <>
      {patches.map((patch) => {
        const isSelected = selectedPatch?.id === patch.id;
        const isBuffering = loadingPatchId === patch.id;

        // --- UPDATED LOGIC ---
        // 1. If global toggle is off, show nothing.
        // 2. If it's the patch currently being analyzed, show buffering color.
        // 3. If the patch has a probability recorded (from AI), show its color.
        const hasResult = patch.probability > 0;
        const shouldShowColor = showProbability && (hasResult || isSelected);

        return (
          <GeoJSON
            // Include patch.probability in the key so Leaflet re-draws when data arrives
            key={`${patch.id}-${isSelected}-${isBuffering}-${patch.probability}`} 
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
              }
            }}
            style={{
              fillColor: isBuffering 
                ? '#f59e0b' 
                : (shouldShowColor ? getProbabilityColor(patch.probability, 0.6) : 'transparent'),
              fillOpacity: isBuffering ? 0.6 : (shouldShowColor ? 0.6 : 0),
              color: isSelected ? '#3b82f6' : (isBuffering ? '#f59e0b' : 'rgba(148, 163, 184, 0.2)'),
              weight: isSelected || isBuffering ? 3 : 0.5,
              dashArray: isBuffering ? "5, 10" : null,
            }}
            eventHandlers={{
              click: () => !loadingPatchId && onPatchClick(patch),
            }}
          >
            {isBuffering && (
              <Tooltip permanent direction="center" sticky>
                <div className="bg-slate-900 text-white px-2 py-1 rounded shadow-lg border border-amber-500">
                  <span className="text-[10px] font-bold text-amber-500 animate-pulse uppercase">
                    AI Processing...
                  </span>
                </div>
              </Tooltip>
            )}
          </GeoJSON>
        );
      })}
    </>
  );
};

export default PatchGridLayer;