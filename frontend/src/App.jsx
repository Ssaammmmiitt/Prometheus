import { BrowserRouter, Route, Routes } from "react-router-dom";
import "leaflet/dist/leaflet.css";

import ApiBanner from "./components/chrome/ApiBanner";
import AppHeader from "./components/chrome/AppHeader";
import DistrictPage from "./pages/DistrictPage";
import FiresPage from "./pages/FiresPage";
import MapPage from "./pages/MapPage";
import PredictPage from "./pages/PredictPage";
import ComparePage from "./pages/ComparePage";
import VerifyPage from "./pages/VerifyPage";
import { ForecastProvider } from "./state/ForecastContext";
import { ThemeProvider } from "./theme/ThemeProvider";

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <ForecastProvider>
          <div className="h-dvh w-full overflow-hidden bg-surface text-ink">
            <AppHeader />
            <ApiBanner />
            <Routes>
              <Route path="/" element={<MapPage />} />
              <Route path="/district/:id" element={<DistrictPage />} />
              <Route path="/predict" element={<PredictPage />} />
              <Route path="/compare" element={<ComparePage />} />
              <Route path="/fires" element={<FiresPage />} />
              <Route path="/verify" element={<VerifyPage />} />
            </Routes>
          </div>
        </ForecastProvider>
      </BrowserRouter>
    </ThemeProvider>
  );
}
