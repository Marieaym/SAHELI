import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Workflow, Map, Bot, FileText, Radio, ChevronRight, AlertTriangle,
  Play, Sparkles, Satellite, Brain, MessageCircle, ShieldCheck,
} from "lucide-react";
import {
  PieChart, Pie, Cell, ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, BarChart, Bar,
} from "recharts";
import {
  getDistricts, getModelMetrics, getFeedEvents,
  getAssistantStatus, getDistrictHistory, getFoodSecurityV2, getKeyMessages,
  getCommandCenterBriefing, RISK_COLORS,
} from "../api/client";
import { LoadingState, LiveDot } from "../components/ui";
import RiskBadge from "../components/RiskBadge";
import { useAuth } from "../context/AuthContext";
import { useLanguage } from "../context/LanguageContext";

const AGENTS = [
  { key: "sentinel", icon: Satellite },
  { key: "forecast", icon: Brain },
  { key: "explainer", icon: MessageCircle },
  { key: "alerter", icon: Radio },
  { key: "policywriter", icon: FileText },
];

function KeyMessages({ data, loading, t, language }) {
  const [expanded, setExpanded] = useState(null);
  const [evidence, setEvidence] = useState({});
  const [evidenceLoading, setEvidenceLoading] = useState(false);

  function toggleBullet(i, district) {
    if (expanded === i) { setExpanded(null); return; }
    setExpanded(i);
    if (!evidence[district]) {
      setEvidenceLoading(true);
      getCommandCenterBriefing(district, language)
        .then((res) => setEvidence((prev) => ({ ...prev, [district]: res })))
        .finally(() => setEvidenceLoading(false));
    }
  }

  if (loading) {
    return (
      <div className="bento-card p-6 mb-5 space-y-3">
        <div className="h-3 w-32 rounded bg-cardBorder/40 animate-pulse-soft" />
        <div className="h-4 w-full rounded bg-cardBorder/40 animate-pulse-soft" />
        <div className="h-4 w-5/6 rounded bg-cardBorder/40 animate-pulse-soft" />
      </div>
    );
  }
  if (!data || !data.bullets?.length) return null;
  const isLive = data.ai_mode?.startsWith("live_");

  return (
    <div className="bento-card p-6 md:p-7 mb-5 relative overflow-hidden">
      <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-primary to-goldBright" />
      <div className="flex items-center justify-between mb-1 pl-2">
        <div className="flex items-center gap-2">
          <Sparkles size={15} className="text-primary" />
          <span className="font-display text-sm font-semibold text-sand uppercase tracking-wide">
            {t("dashboard.keyMessages")}
          </span>
        </div>
        <span className={`font-mono text-[9px] px-1.5 py-0.5 rounded-[2px] uppercase tracking-wide ${
          isLive ? "text-acacia border border-acacia/40 bg-acacia/10" : "text-muted border border-cardBorder bg-surface"
        }`}>
          {isLive ? t("aiNarrative.live") : t("aiNarrative.summary")}
        </span>
      </div>
      <p className="text-muted text-[10px] pl-2 mb-3 italic">{t("dashboard.keyMessagesHint")}</p>

      <ul className="space-y-1 pl-2">
        {data.bullets.map((bullet, i) => {
          const parts = bullet.text.split(/\*\*(.+?)\*\*/g);
          const clickable = bullet.districts?.length > 0;
          const district = bullet.districts?.[0];
          const isOpen = expanded === i;
          const ev = district ? evidence[district] : null;

          return (
            <li key={i}>
              <button
                onClick={() => clickable && toggleBullet(i, district)}
                className={`w-full text-left flex items-start gap-2.5 text-sm leading-relaxed py-1.5 rounded-lg transition-colors ${
                  clickable ? "hover:bg-primary/5 cursor-pointer" : "cursor-default"
                }`}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5 flex-shrink-0" />
                <span className="text-sand/90 flex-1">
                  {parts.map((part, j) =>
                    j % 2 === 1 ? <b key={j} className="text-sand font-semibold">{part}</b> : <span key={j}>{part}</span>
                  )}
                </span>
                {clickable && (
                  <ChevronRight size={14} className={`text-primary flex-shrink-0 mt-1 transition-transform ${isOpen ? "rotate-90" : ""}`} />
                )}
              </button>

              {isOpen && (
                <div className="ml-4 mt-1 mb-3 pl-3 border-l-2 border-primary/30 animate-fade-up">
                  {evidenceLoading && !ev ? (
                    <div className="flex items-center gap-2 text-muted text-xs py-2">
                      <Sparkles size={12} className="animate-pulse-soft" /> {t("dashboard.loadingEvidence")}
                    </div>
                  ) : ev ? (
                    <div className="bg-surface/60 border border-cardBorder rounded-xl p-3.5">
                      <div className="text-[10px] uppercase tracking-wide text-muted font-mono mb-2">
                        {t("dashboard.liveEvidenceFor")} {district}
                      </div>
                      <div className="flex flex-wrap gap-3 text-xs mb-2.5">
                        <span className="text-muted">{t("scenario.climateShock")}: <RiskBadge level={ev.signals.climate_risk} size="sm" /></span>
                        {ev.signals.food_security && (
                          <span className="text-muted">{t("scenario.foodSecurity")}: <RiskBadge level={ev.signals.food_security.risk} size="sm" /></span>
                        )}
                      </div>
                      <p className="text-sand/80 text-xs leading-relaxed mb-3">{ev.briefing}</p>
                      <div className="flex items-center gap-3">
                        <Link to={`/command-center`} className="text-primary text-xs font-medium inline-flex items-center gap-1">
                          {t("dashboard.fullCommandCenter")} <ChevronRight size={12} />
                        </Link>
                        <Link to={`/scenario`} className="text-acacia text-xs font-medium inline-flex items-center gap-1">
                          <ShieldCheck size={12} /> {t("dashboard.testScenario")} {district}
                        </Link>
                      </div>
                    </div>
                  ) : null}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function LiveClock({ className = "" }) {
  const { language } = useLanguage();
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <div className={className}>
      <div className="font-display font-bold text-4xl text-white tracking-tight">
        {now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
      </div>
      <div className="text-sm text-white/75 mt-1">
        {now.toLocaleDateString(language === "fr" ? "fr-FR" : "en-US", { weekday: "long", month: "long", day: "numeric" })}
      </div>
    </div>
  );
}

function AiStatusBadge({ ai, t }) {
  if (ai.live_ok) return <span className="ai-badge">{t("dashboard.gptLive")}</span>;
  if (ai.error_code === "quota_exceeded") {
    return <span className="stat-pill bg-clay/15 text-clay border border-clay/25 text-[10px]">{t("dashboard.aiQuota")}</span>;
  }
  if (ai.ready) {
    return <span className="stat-pill bg-primary/10 text-primary border border-primary/20 text-[10px]">{t("dashboard.aiDataMode")}</span>;
  }
  return <span className="stat-pill bg-muted/10 text-muted border border-cardBorder text-[10px]">{t("topbar.aiOffline")}</span>;
}

export default function Overview() {
  const { user } = useAuth();
  const { t, language } = useLanguage();
  const [districts, setDistricts] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [zones, setZones] = useState([]);
  const [feed, setFeed] = useState([]);
  const [trend, setTrend] = useState([]);
  const [ai, setAi] = useState({ ready: false });
  const [error, setError] = useState(null);
  const [topV2, setTopV2] = useState(null);
  const [keyMessages, setKeyMessages] = useState(null);
  const [keyMessagesLoading, setKeyMessagesLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getDistricts(),
      getModelMetrics(),
      getFeedEvents(5).catch(() => ({ events: [] })),
      getAssistantStatus().catch(() => ({ ready: false })),
    ])
      .then(async ([d, m, f, aiStatus]) => {
        setDistricts(d.districts);
        setMetrics(m);
        const zoneMap = {};
        d.districts.forEach((dist) => {
          if (!zoneMap[dist.zone]) zoneMap[dist.zone] = { zone: dist.zone, n: 0, drought: 0 };
          zoneMap[dist.zone].n += 1;
          zoneMap[dist.zone].drought += dist.drought_index;
        });
        setZones(Object.values(zoneMap).map((z) => ({
          zone: z.zone,
          avg_drought_index: Number((z.drought / z.n).toFixed(2)),
        })));
        setFeed(f.events || []);
        setAi(aiStatus);
        const top = [...d.districts].sort(
          (a, b) => ({ Critical: 0, High: 1, Medium: 2, Low: 3 }[a.predicted_risk] - { Critical: 0, High: 1, Medium: 2, Low: 3 }[b.predicted_risk])
        )[0];
        if (top) {
          const h = await getDistrictHistory(top.district, 90).catch(() => ({ history: [] }));
          setTrend((h.history || []).map((r) => ({ date: r.date.slice(5), drought: r.drought_index })));
          getFoodSecurityV2(top.district).then(setTopV2).catch(() => setTopV2(null));
        }
      })
      .catch((e) => setError(e.message));

    getKeyMessages(language)
      .then(setKeyMessages)
      .catch(() => setKeyMessages(null))
      .finally(() => setKeyMessagesLoading(false));
  }, [language]);

  const riskPie = useMemo(() => {
    if (!districts) return [];
    return ["Critical", "High", "Medium", "Low"].map((level) => ({
      name: level,
      value: districts.filter((d) => d.predicted_risk === level).length,
      fill: RISK_COLORS[level],
    })).filter((d) => d.value > 0);
  }, [districts]);

  if (error) {
    return <div className="bento-card p-8 text-clay">{t("overview.apiError")} — {error}</div>;
  }
  if (!districts || !metrics) return <LoadingState message={t("overview.loading")} />;

  const nCritical = districts.filter((d) => d.predicted_risk === "Critical").length;
  const nHigh = districts.filter((d) => d.predicted_risk === "High").length;
  const topCritical = districts.filter((d) => d.predicted_risk === "Critical").slice(0, 3);
  const firstName = user?.full_name?.split(" ")[0] || "Agent";
  const topDistrict = [...districts].sort(
    (a, b) => ({ Critical: 0, High: 1, Medium: 2, Low: 3 }[a.predicted_risk] - { Critical: 0, High: 1, Medium: 2, Low: 3 }[b.predicted_risk])
  )[0];

  const modules = [
    { to: "/pipeline", icon: Workflow, label: t("nav.agentPipeline"), color: "from-indigo-900 to-indigo-700" },
    { to: "/map", icon: Map, label: t("nav.riskMap"), color: "from-amber to-clay" },
    { to: "/brief", icon: FileText, label: t("nav.policyBrief"), color: "from-acacia to-emerald-800" },
    { to: "/scenario", icon: ShieldCheck, label: t("nav.scenarioSimulator"), color: "from-slate-700 to-slate-900" },
  ];

  return (
    <div className="animate-fade-up dash-grid">
      <KeyMessages data={keyMessages} loading={keyMessagesLoading} t={t} language={language} />

      {/* ── HERO BENTO ── */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 md:gap-5">
        <div className="xl:col-span-8 bento-hero p-6 md:p-8 min-h-[300px] flex flex-col justify-between">
          <div className="z-10">
              <div className="flex flex-wrap items-center gap-2 mb-3">
                <LiveDot label={t("common.live")} />
                <span className="text-white/50 text-[10px] font-mono uppercase tracking-widest">{user?.country}</span>
              </div>
              <h1 className="font-display font-bold text-3xl md:text-[2.4rem] text-white leading-tight">
                {t("dashboard.welcome")}, {firstName}
              </h1>
              <p className="text-white/70 text-sm mt-2 max-w-lg leading-relaxed">
                {t("dashboard.summary").replace("{critical}", nCritical).replace("{total}", districts.length)}
              </p>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-6">
              {[
                { label: t("dashboard.criticalMetric"), value: nCritical, sub: `/${districts.length}` },
                { label: t("dashboard.highRisk"), value: nHigh },
                { label: "F1", value: `${(metrics.weighted_f1 * 100).toFixed(0)}%` },
                { label: t("overview.agents"), value: "5" },
              ].map(({ label, value, sub }) => (
                <div key={label} className="glass-dark px-3 py-3 rounded-xl">
                  <div className="text-[9px] uppercase tracking-widest text-white/50 font-mono">{label}</div>
                  <div className="font-display font-bold text-2xl text-white mt-0.5">
                    {value}{sub && <span className="text-sm text-white/50">{sub}</span>}
                  </div>
                </div>
              ))}
            </div>

            {topDistrict && (
              <div className="glass-dark p-4 mt-4 flex items-start gap-3">
                <AlertTriangle size={18} className="text-amber-300 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <div className="text-white text-sm font-medium">{t("dashboard.predictionTitle")}</div>
                  <p className="text-white text-base font-display font-bold mt-1">
                    {topDistrict.district}, {topDistrict.country}
                  </p>
                  <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-1.5 text-xs">
                    <span className="text-white/70">
                      {t("dashboard.climateLabel")}: <b className="text-white">{topDistrict.predicted_risk}</b>
                      {" "}({t("dashboard.droughtIdxLabel")} {topDistrict.drought_index?.toFixed(2)})
                    </span>
                    {topV2 && (
                      <span className="text-white/70">
                        {t("dashboard.foodSecLabel")}: <b className="text-white">{topV2.predicted_risk_level}</b>
                        {" "}(IPC {topV2.predicted_ipc_phase?.toFixed(2)}/5, {topV2.ground_truth_status === "validated" ? t("dashboard.fsValidated") : t("dashboard.fsExtrapolated")})
                      </span>
                    )}
                  </div>
                  {topCritical.length > 1 && (
                    <p className="text-white/60 text-xs mt-2">
                      {t("dashboard.alsoWatching")} {topCritical.filter((d) => d.district !== topDistrict.district).map((d) => d.district).join(", ")}
                    </p>
                  )}
                  <Link to="/map" className="btn-glass mt-2 text-xs inline-flex">{t("dashboard.viewMap")} <ChevronRight size={12} /></Link>
                </div>
              </div>
            )}

            <Link to="/pipeline" className="btn-glass w-fit mt-4"><Play size={14} /> {t("dashboard.runPipeline")}</Link>
        </div>

        <div className="xl:col-span-4 flex flex-col gap-4 md:gap-5">
          <div className="bento-photo p-5 flex-1 min-h-[160px] relative z-0">
            <LiveClock />
          </div>
          <div className="bento-card p-5 flex-1">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Bot size={17} className="text-primary" />
                <span className="font-display font-semibold text-sand text-sm">{t("nav.aiAssistant")}</span>
              </div>
              <AiStatusBadge ai={ai} t={t} />
            </div>
            <p className="text-xs text-muted leading-relaxed mt-1">
              {ai.live_ok ? t("dashboard.aiTeaser")
                : ai.error_code === "quota_exceeded" ? t("dashboard.aiQuotaHint")
                : ai.ready ? t("dashboard.aiDataModeHint")
                : t("dashboard.aiOfflineHint")}
            </p>
            <Link to="/assistant" className="inline-flex items-center gap-1 text-sm text-primary font-medium mt-3">
              <Sparkles size={14} /> {t("dashboard.askNow")} <ChevronRight size={14} />
            </Link>
          </div>
        </div>
      </div>

      {/* ── ANALYTICS ROW ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-12 gap-4 md:gap-5">
        <div className="xl:col-span-3 bento-card p-5">
          <div className="text-[10px] uppercase tracking-widest text-muted font-mono mb-1">{t("overview.riskMix")}</div>
          <ResponsiveContainer width="100%" height={160}>
            <PieChart>
              <Pie data={riskPie} innerRadius={42} outerRadius={68} paddingAngle={3} dataKey="value">
                {riskPie.map((e) => <Cell key={e.name} fill={e.fill} />)}
              </Pie>
              <Tooltip contentStyle={{ background: "rgb(var(--c-card))", border: "1px solid rgb(var(--c-cardBorder))", borderRadius: 10, fontSize: 11 }} />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-2 mt-1">
            {riskPie.map((r) => (
              <span key={r.name} className="text-[10px] text-muted flex items-center gap-1">
                <span className="w-2 h-2 rounded-full" style={{ background: r.fill }} /> {r.name} ({r.value})
              </span>
            ))}
          </div>
        </div>

        <div className="xl:col-span-5 bento-card p-5">
          <div className="flex items-center justify-between mb-2">
            <div>
              <div className="text-[10px] uppercase tracking-widest text-muted font-mono">{t("overview.droughtTrend")}</div>
              <div className="font-display font-semibold text-sand text-sm">{topDistrict?.district}</div>
            </div>
            <RiskBadge level={topDistrict?.predicted_risk} size="sm" />
          </div>
          <ResponsiveContainer width="100%" height={150}>
            <AreaChart data={trend}>
              <defs>
                <linearGradient id="droughtFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#d6a24a" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#d6a24a" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="date" tick={{ fontSize: 9, fill: "rgb(var(--c-muted))" }} interval="preserveStartEnd" />
              <YAxis hide domain={["auto", "auto"]} />
              <Tooltip contentStyle={{ background: "rgb(var(--c-card))", border: "1px solid rgb(var(--c-cardBorder))", borderRadius: 10, fontSize: 11 }} />
              <Area type="monotone" dataKey="drought" stroke="#d6a24a" fill="url(#droughtFill)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="xl:col-span-4 bento-card p-5">
          <div className="text-[10px] uppercase tracking-widest text-muted font-mono mb-3">{t("overview.agentStack")}</div>
          <div className="space-y-2">
            {AGENTS.map(({ key, icon: Icon }, i) => (
              <div key={key} className="flex items-center gap-3 p-2.5 rounded-xl bg-surface/60 border border-cardBorder/50">
                <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                  <Icon size={15} className="text-primary" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium text-sand truncate">{t(`pipeline.agents.${key}.name`)}</div>
                  <div className="text-[10px] text-muted truncate">{t(`pipeline.agents.${key}.role`)}</div>
                </div>
                <span className="font-mono text-[10px] text-muted">{String(i + 1).padStart(2, "0")}</span>
              </div>
            ))}
          </div>
          <Link to="/pipeline" className="text-primary text-xs font-medium inline-flex items-center gap-1 mt-3">
            {t("dashboard.launch")} <ChevronRight size={12} />
          </Link>
        </div>
      </div>

      {/* ── ZONES + MODULES ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 md:gap-5">
        <div className="lg:col-span-4 bento-card p-5">
          <div className="text-[10px] uppercase tracking-widest text-muted font-mono mb-3">{t("overview.zoneBreakdown")}</div>
          <ResponsiveContainer width="100%" height={zones.length * 36 + 20}>
            <BarChart data={zones} layout="vertical" margin={{ left: 0, right: 8 }}>
              <XAxis type="number" hide />
              <YAxis type="category" dataKey="zone" width={72} tick={{ fontSize: 10, fill: "rgb(var(--c-muted))" }} />
              <Tooltip contentStyle={{ background: "rgb(var(--c-card))", border: "1px solid rgb(var(--c-cardBorder))", borderRadius: 10, fontSize: 11 }} />
              <Bar dataKey="avg_drought_index" fill="#d6a24a" radius={[0, 6, 6, 0]} barSize={14} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="lg:col-span-8 grid grid-cols-2 md:grid-cols-4 gap-3">
          {modules.map(({ to, icon: Icon, label, color }) => (
            <Link key={to} to={to} className="bento-card p-0 overflow-hidden group hover:-translate-y-0.5 transition-transform">
              <div className={`h-20 bg-gradient-to-br ${color} flex items-center justify-center`}>
                <Icon size={28} className="text-white/90 group-hover:scale-110 transition-transform" />
              </div>
              <div className="p-3">
                <div className="text-xs font-display font-semibold text-sand leading-tight">{label}</div>
                <ChevronRight size={14} className="text-primary mt-2 opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* ── FEED + RANKING ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 md:gap-5">
        <div className="bento-card p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-[10px] uppercase tracking-widest text-muted font-mono">{t("nav.liveFeed")}</div>
              <h3 className="font-display font-semibold text-lg text-sand">{t("dashboard.recentEvents")}</h3>
            </div>
            <Link to="/feed" className="text-primary text-xs font-medium">{t("dashboard.viewAll")}</Link>
          </div>
          <div className="divide-y divide-cardBorder">
            {(feed.length ? feed : districts.slice(0, 5).map((d) => ({
              district: d.district, from_risk: "Medium", to_risk: d.predicted_risk,
            }))).slice(0, 5).map((ev, i) => (
              <div key={i} className="flex items-center justify-between py-3 first:pt-0">
                <div>
                  <div className="text-sm font-medium text-sand">{ev.district}</div>
                  <div className="text-xs text-muted">{ev.from_risk} → {ev.to_risk}</div>
                </div>
                <RiskBadge level={ev.to_risk} size="sm" />
              </div>
            ))}
          </div>
        </div>

        <div className="bento-card p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-[10px] uppercase tracking-widest text-muted font-mono">{t("dashboard.rankingLabel")}</div>
              <h3 className="font-display font-semibold text-lg text-sand">{t("dashboard.topRisk")}</h3>
            </div>
            <Link to="/compare" className="text-primary text-xs font-medium">{t("nav.compareDistricts")}</Link>
          </div>
          <div className="space-y-1">
            {[...districts]
              .sort((a, b) => ({ Critical: 0, High: 1, Medium: 2, Low: 3 }[a.predicted_risk] - { Critical: 0, High: 1, Medium: 2, Low: 3 }[b.predicted_risk]))
              .slice(0, 6)
              .map((d, i) => (
                <Link key={d.district} to={`/map?district=${encodeURIComponent(d.district)}`}
                  className="flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-night/5 transition-colors group">
                  <span className="font-mono text-xs text-muted w-5">{String(i + 1).padStart(2, "0")}</span>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-sand truncate">{d.district}</div>
                    <div className="text-[11px] text-muted">{d.zone}</div>
                  </div>
                  <RiskBadge level={d.predicted_risk} size="sm" />
                  <ChevronRight size={14} className="text-muted opacity-0 group-hover:opacity-100" />
                </Link>
              ))}
          </div>
        </div>
      </div>
    </div>
  );
}
