import { NavLink, useNavigate, useLocation } from "react-router-dom";
import { useMemo } from "react";
import {
  Home, Workflow, Map, Radio, GitCompare, Waves, Sliders, FileText,
  MessageSquare, Bot, GitBranch, ShieldCheck, ChevronRight, LogOut, Bell, Sun, Moon, Leaf, Crosshair, X,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { useLanguage } from "../context/LanguageContext";
import { useTheme } from "../context/ThemeContext";
import SaheliLogo from "./SaheliLogo";

const NAV_GROUPS = [
  {
    labelKey: "nav.groupOverview",
    items: [
      { to: "/command-center", labelKey: "nav.commandCenter", icon: Crosshair },
      { to: "/", labelKey: "nav.dashboard", icon: Home },
      { to: "/pipeline", labelKey: "nav.agentPipeline", icon: Workflow },
    ],
  },
  {
    labelKey: "nav.groupMonitor",
    items: [
      { to: "/map", labelKey: "nav.riskMap", icon: Map },
      { to: "/feed", labelKey: "nav.liveFeed", icon: Radio },
      { to: "/compare", labelKey: "nav.compareDistricts", icon: GitCompare },
      { to: "/cv-scanner", labelKey: "nav.cornScanner", icon: Leaf },
    ],
  },
  {
    labelKey: "nav.groupSimulate",
    items: [
      { to: "/scenario", labelKey: "nav.scenarioSimulator", icon: Waves },
      { to: "/intervention", labelKey: "nav.interventionSimulator", icon: Sliders },
    ],
  },
  {
    labelKey: "nav.groupAct",
    items: [
      { to: "/brief", labelKey: "nav.policyBrief", icon: FileText },
      { to: "/alerts", labelKey: "nav.alertSimulator", icon: MessageSquare },
      { to: "/assistant", labelKey: "nav.aiAssistant", icon: Bot, ai: true },
    ],
  },
  {
    labelKey: "nav.groupTrust",
    items: [
      { to: "/causal", labelKey: "nav.causalPathway", icon: GitBranch },
      { to: "/validation", labelKey: "nav.modelValidation", icon: ShieldCheck },
    ],
  },
];

export default function Sidebar({ isOpen = false, onClose = () => {} }) {
  const { user, logout } = useAuth();
  const { language, setLanguage, t } = useLanguage();
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();

  const groups = useMemo(() => {
    // Role-based nav filtering removed along with the "Viewing as" switcher —
    // every authenticated user now sees the full nav for their country scope.
    return NAV_GROUPS.filter((g) => g.items.length > 0);
  }, []);

  const initials = user?.full_name?.split(" ").map((n) => n[0]).join("").slice(0, 2).toUpperCase() || "?";

  return (
    <>
      {/* Backdrop, mobile only: click outside the drawer to close it */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-night/60 z-40 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`sidebar-panel fixed inset-y-0 left-0 z-50 transition-transform duration-300 ease-in-out
          ${isOpen ? "translate-x-0" : "-translate-x-full"} lg:translate-x-0 lg:static lg:z-auto`}
      >
        <button
          type="button"
          onClick={onClose}
          className="lg:hidden absolute top-4 right-3 btn-ghost p-1.5"
          aria-label={t("nav.closeMenu")}
        >
          <X size={18} />
        </button>

        <div className="px-4 pt-5 pb-3 border-b border-cardBorder/60">
          <SaheliLogo />
        </div>

      <div className="flex items-center gap-1.5 px-4 py-2.5 border-b border-cardBorder/60">
        <span className="stat-pill bg-acacia/10 text-acacia border border-acacia/20 text-[9px]">
          ● {t("common.live")}
        </span>
        <div className="flex rounded-lg overflow-hidden border border-cardBorder text-[9px] font-mono ml-auto">
          {["en", "fr"].map((lng) => (
            <button
              key={lng}
              onClick={() => setLanguage(lng)}
              className={`px-2 py-1 transition-colors ${language === lng ? "bg-primary text-night font-semibold" : "text-muted hover:text-sand"}`}
            >
              {lng.toUpperCase()}
            </button>
          ))}
        </div>
        <button onClick={toggleTheme} className="btn-ghost p-1.5" aria-label={t("nav.toggleTheme")}>
          {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
        </button>
      </div>

      <nav className="flex-1 px-2 py-2 overflow-y-auto">
        {groups.map(({ labelKey, items }) => (
          <div key={labelKey}>
            <div className="nav-section-label">{t(labelKey)}</div>
            <div className="space-y-0.5 pb-1">
              {items.map(({ to, labelKey, icon: Icon, ai }) => {
                const active = location.pathname === to || (to !== "/" && location.pathname.startsWith(to));
                return (
                  <NavLink key={to} to={to} onClick={onClose} className={`nav-pill ${active ? "nav-pill-active" : ""}`}>
                    <Icon size={17} strokeWidth={active ? 2.2 : 1.8} />
                    <span className="flex-1 truncate">{t(labelKey)}</span>
                    {ai && !active && <span className="ai-badge">AI</span>}
                    {active && <ChevronRight size={14} className="opacity-70 flex-shrink-0" />}
                  </NavLink>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="px-3 py-3 space-y-2.5 border-t border-cardBorder">
        {user && (
          <button
            onClick={() => navigate("/profile")}
            className="w-full flex items-center gap-2.5 px-2 py-2.5 rounded-xl bg-night/5 border border-cardBorder/60 hover:border-primary/40 transition-colors text-left"
          >
            {user.photo_base64 ? (
              <img src={user.photo_base64} alt="" className="w-9 h-9 rounded-full object-cover flex-shrink-0" />
            ) : (
              <div className="w-9 h-9 rounded-full bg-gradient-to-br from-gold to-amber flex items-center justify-center text-night text-xs font-bold flex-shrink-0">
                {initials}
              </div>
            )}
            <div className="min-w-0 flex-1">
              <div className="text-xs font-semibold text-sand truncate">{user.full_name}</div>
              <div className="text-[10px] text-muted truncate">{user.country}</div>
            </div>
          </button>
        )}

        <div className="flex gap-1">
          <button className="nav-pill flex-1 text-[12px] opacity-70 py-2">
            <Bell size={14} /> <span className="truncate">{t("nav.notifications")}</span>
          </button>
          <button
            onClick={() => { logout(); navigate("/login"); }}
            className="nav-pill text-clay hover:bg-clay/5 py-2 px-3"
            title={t("nav.signOut")}
          >
            <LogOut size={15} />
          </button>
        </div>
      </div>
    </aside>
    </>
  );
}
