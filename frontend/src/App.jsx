import { useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { RoleProvider } from "./context/RoleContext";
import { AuthProvider } from "./context/AuthContext";
import { LanguageProvider } from "./context/LanguageContext";
import { ThemeProvider } from "./context/ThemeContext";
import ProtectedRoute from "./components/ProtectedRoute";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import FloatingAssistant from "./components/FloatingAssistant";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Overview from "./pages/Overview";
import AgentPipeline from "./pages/AgentPipeline";
import CommandCenter from "./pages/CommandCenter";
import RiskMap from "./pages/RiskMap";
import PolicyBrief from "./pages/PolicyBrief";
import AlertSimulator from "./pages/AlertSimulator";
import InterventionSimulator from "./pages/InterventionSimulator";
import Assistant from "./pages/Assistant";
import LiveFeed from "./pages/LiveFeed";
import ScenarioSimulator from "./pages/ScenarioSimulator";
import CompareDistricts from "./pages/CompareDistricts";
import CausalPathway from "./pages/CausalPathway";
import ModelValidation from "./pages/ModelValidation";
import Profile from "./pages/Profile";
import CornScanner from "./pages/CornScanner";
import NotFound from "./pages/NotFound";

function DashboardLayout({ children }) {
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  return (
    <div className="app-shell">
      <Sidebar isOpen={mobileSidebarOpen} onClose={() => setMobileSidebarOpen(false)} />
      <div className="flex-1 min-w-0 flex flex-col min-h-screen">
        <TopBar onMenuClick={() => setMobileSidebarOpen(true)} />
        <main className="page-shell flex-1">{children}</main>
      </div>
      <FloatingAssistant />
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
    <AuthProvider>
      <LanguageProvider>
      <RoleProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/*" element={
              <ProtectedRoute>
                <DashboardLayout>
                  <Routes>
                    <Route path="/" element={<Overview />} />
                    <Route path="/pipeline" element={<AgentPipeline />} />
                    <Route path="/command-center" element={<CommandCenter />} />
                    <Route path="/map" element={<RiskMap />} />
                    <Route path="/feed" element={<LiveFeed />} />
                    <Route path="/scenario" element={<ScenarioSimulator />} />
                    <Route path="/compare" element={<CompareDistricts />} />
                    <Route path="/causal" element={<CausalPathway />} />
                    <Route path="/validation" element={<ModelValidation />} />
                    <Route path="/brief" element={<PolicyBrief />} />
                    <Route path="/alerts" element={<AlertSimulator />} />
                    <Route path="/intervention" element={<InterventionSimulator />} />
                    <Route path="/assistant" element={<Assistant />} />
                    <Route path="/messages" element={<Assistant />} />
                    <Route path="/profile" element={<Profile />} />
                    <Route path="/cv-scanner" element={<CornScanner />} />
                    <Route path="*" element={<NotFound />} />
                  </Routes>
                </DashboardLayout>
              </ProtectedRoute>
            } />
          </Routes>
        </BrowserRouter>
      </RoleProvider>
      </LanguageProvider>
    </AuthProvider>
    </ThemeProvider>
  );
}
