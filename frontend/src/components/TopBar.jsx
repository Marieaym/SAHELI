import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Search, Bell, Bot, Menu } from "lucide-react";
import { getDistricts, getAssistantStatus } from "../api/client";
import { useAuth } from "../context/AuthContext";
import { useLanguage } from "../context/LanguageContext";

function aiStatusLabel(ai, t) {
  if (ai.live_ok) return { label: t("dashboard.gptLive"), cls: "bg-acacia/10 text-acacia border-acacia/25" };
  if (ai.error_code === "quota_exceeded") return { label: t("topbar.aiQuota"), cls: "bg-clay/10 text-clay border-clay/25" };
  if (ai.ready) return { label: t("topbar.aiDataMode"), cls: "bg-primary/10 text-primary border-primary/25" };
  return { label: t("topbar.aiOffline"), cls: "bg-amber/10 text-amber border-amber/25" };
}

export default function TopBar({ onMenuClick }) {
  const { user } = useAuth();
  const { t, language } = useLanguage();
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [districts, setDistricts] = useState([]);
  const [open, setOpen] = useState(false);
  const [ai, setAi] = useState({ ready: false, live_ok: false, error_code: null });

  useEffect(() => {
    getDistricts().then((d) => setDistricts(d.districts || [])).catch(() => {});
    getAssistantStatus().then(setAi).catch(() => setAi({ ready: false, live_ok: false }));
  }, []);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    return districts.filter((d) => d.district.toLowerCase().includes(q)).slice(0, 6);
  }, [query, districts]);

  const now = new Date();
  const time = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const date = now.toLocaleDateString(language === "fr" ? "fr-FR" : "en-US", {
    weekday: "short", month: "short", day: "numeric",
  });

  const { label: aiLabel, cls: aiClass } = aiStatusLabel(ai, t);

  return (
    <header className="top-bar">
      <button
        type="button"
        onClick={onMenuClick}
        className="btn-ghost p-2 lg:hidden flex-shrink-0"
        aria-label={t("nav.openMenu")}
      >
        <Menu size={20} />
      </button>

      <div className="relative flex-1 max-w-md">
        <Search size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-muted" />
        <input
          value={query}
          onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          placeholder={t("nav.jumpToDistrict")}
          className="input-field pl-10 py-2.5 text-sm bg-card/80"
        />
        {open && matches.length > 0 && (
          <div className="absolute z-50 top-full mt-1 w-full bento-card py-1 shadow-bento-lg">
            {matches.map((d) => (
              <button
                key={d.district}
                type="button"
                onMouseDown={() => navigate(`/map?district=${encodeURIComponent(d.district)}`)}
                className="w-full text-left px-4 py-2.5 text-sm hover:bg-night/5 flex justify-between gap-2"
              >
                <span className="text-sand font-medium">{d.district}</span>
                <span className="text-muted text-xs">{d.predicted_risk}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="flex items-center gap-3 ml-auto">
        <span className={`stat-pill border text-[10px] hidden sm:inline-flex ${aiClass}`}>
          <Bot size={12} /> {aiLabel}
        </span>
        <div className="text-right hidden md:block">
          <div className="font-mono text-sm font-semibold text-sand">{time}</div>
          <div className="text-[10px] text-muted uppercase tracking-wide">{date}</div>
        </div>
        <button type="button" className="btn-ghost p-2 relative" aria-label={t("nav.notifications")}>
          <Bell size={18} />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-clay" />
        </button>
        {user && (
          <div className="hidden lg:flex items-center gap-2 pl-3 border-l border-cardBorder">
            <div className="text-right">
              <div className="text-xs font-semibold text-sand">{user.full_name}</div>
              <div className="text-[10px] text-muted">{user.country}</div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}
