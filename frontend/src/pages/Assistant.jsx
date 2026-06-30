import { useEffect, useRef, useState } from "react";
import { Send, Bot, User, Loader2, TrendingUp, Sparkles } from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { getDistricts, askAssistant, getDistrictHistory, getAssistantStatus } from "../api/client";
import { PageHeader, BentoCard, LoadingState, SectionLabel, AiBadge, LiveDot } from "../components/ui";
import RiskBadge from "../components/RiskBadge";
import { useAuth } from "../context/AuthContext";
import { useLanguage } from "../context/LanguageContext";

const SEVERITY_ORDER = { Critical: 0, High: 1, Medium: 2, Low: 3 };

export default function Assistant() {
  const { user } = useAuth();
  const { t, language } = useLanguage();
  const [districts, setDistricts] = useState(null);
  const [district, setDistrict] = useState("");
  const [messages, setMessages] = useState([]);
  const [briefingLoading, setBriefingLoading] = useState(true);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState(null);
  const [trend, setTrend] = useState(null);
  const [ai, setAi] = useState(null);
  const scrollRef = useRef(null);
  const briefingRan = useRef(false);

  const country = user?.country || (language === "fr" ? "votre pays" : "your country");
  const SUGGESTED = [
    t("assistant.suggested1").replace("{country}", country),
    t("assistant.suggested2"),
    t("assistant.suggested3"),
  ];

  useEffect(() => {
    getAssistantStatus().then(setAi).catch(() => setAi({ openai_configured: false, live_ok: false }));
    getDistricts().then((d) => {
      const sorted = [...d.districts].sort(
        (a, b) => SEVERITY_ORDER[a.predicted_risk] - SEVERITY_ORDER[b.predicted_risk]
      );
      setDistricts(sorted);
      if (sorted.length) {
        getDistrictHistory(sorted[0].district, 120).then((h) =>
          setTrend(h.history.map((r) => ({ date: r.date.slice(5, 10), drought: r.drought_index })))
        );
      }
    });
  }, []);

  useEffect(() => {
    if (briefingRan.current) return;
    briefingRan.current = true;
    setBriefingLoading(true);
    askAssistant(
      language === "fr"
        ? `Donne un briefing de bienvenue bref (2-3 phrases) sur la situation alimentaire actuelle en ${user?.country || "le pays"}.`
        : `Give a brief, 2-3 sentence welcome briefing on the current food-security situation in ${user?.country || "the country"}.`,
      null,
      language
    )
      .then((res) => {
        setMessages([{ role: "assistant", text: res.answer, mode: res.mode, errorCode: res.error_code }]);
        setMode(res.mode);
        if (res.error_code) getAssistantStatus().then(setAi).catch(() => {});
      })
      .catch(() => {
        setMessages([{
          role: "assistant",
          text: language === "fr"
            ? "Je suis l'Agent Explainer SAHELI. Posez-moi des questions sur les conditions de risque alimentaire dans votre pays."
            : "I'm the SAHELI Agent Explainer. Ask me about current food-security risk conditions for any district in your country.",
          mode: "fallback",
        }]);
      })
      .finally(() => setBriefingLoading(false));
  }, [user, language]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function send(text) {
    const question = text ?? input;
    if (!question.trim()) return;
    setMessages((m) => [...m, { role: "user", text: question }]);
    setInput("");
    setLoading(true);
    try {
      const res = await askAssistant(question, district || null, language);
      setMode(res.mode);
      if (res.error_code) getAssistantStatus().then(setAi).catch(() => {});
      setMessages((m) => [...m, { role: "assistant", text: res.answer, mode: res.mode, aiError: res.ai_error, errorCode: res.error_code }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", text: `${t("overview.apiError")}: ${e.message}`, mode: "error" }]);
    } finally {
      setLoading(false);
    }
  }

  function focusDistrict(name) {
    setDistrict(name);
    getDistrictHistory(name, 120).then((h) =>
      setTrend(h.history.map((r) => ({ date: r.date.slice(5, 10), drought: r.drought_index })))
    );
  }

  const counts = districtCounts(districts);

  if (!districts) return <LoadingState message={t("overview.loading")} />;

  const modeBadge = mode?.startsWith("live_") ? t("assistant.liveGpt")
    : mode === "indicator_summary" ? t("assistant.dataModeNote")
    : (mode === "fallback_error" || ai?.error_code === "quota_exceeded") ? t("assistant.quotaNote")
    : null;

  const statusPill = ai?.live_ok
    ? <LiveDot label={t("assistant.liveAi")} />
    : ai?.error_code === "quota_exceeded"
      ? <span className="stat-pill text-clay border-clay/30 bg-clay/10 text-[10px]">{t("dashboard.aiQuota")}</span>
      : ai?.ready
        ? <span className="stat-pill text-primary border-primary/30 bg-primary/10 text-[10px]">{t("topbar.aiDataMode")}</span>
        : <span className="stat-pill text-amber border-amber/30 bg-amber/10 text-[10px]">{t("assistant.offlineAi")}</span>;

  return (
    <div className="animate-fade-up">
      <PageHeader
        eyebrow={t("assistant.eyebrow")}
        title={t("assistant.title")}
        subtitle={t("assistant.subtitle")}
        badge={
          <div className="flex items-center gap-2">
            <AiBadge />
            {statusPill}
          </div>
        }
      />

      {modeBadge && (
        <div className={`mb-5 bento-card px-4 py-3 text-sm leading-relaxed ${
          mode?.startsWith("live_")
            ? "border-acacia/30 bg-acacia/5 text-sand"
            : "border-clay/30 bg-clay/5 text-sand"
        }`}>
          <Sparkles size={14} className="inline mr-2 text-primary" />
          {modeBadge}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-5">
        <div className="lg:col-span-1 space-y-4">
          <BentoCard>
            <SectionLabel>{t("assistant.countryPulse")}</SectionLabel>
            <div className="grid grid-cols-2 gap-2">
              {["Critical", "High", "Medium", "Low"].map((lvl) => (
                <div key={lvl} className="text-center bg-surface/60 rounded-2xl py-3 border border-cardBorder/50">
                  <div className="font-display text-2xl text-sand">{counts?.[lvl] ?? 0}</div>
                  <RiskBadge level={lvl} size="sm" />
                </div>
              ))}
            </div>
          </BentoCard>

          <BentoCard>
            <SectionLabel>{t("assistant.rankedDistricts")}</SectionLabel>
            <div className="space-y-1 max-h-[200px] overflow-y-auto pr-1">
              {districts.map((d) => (
                <button
                  key={d.district}
                  onClick={() => focusDistrict(d.district)}
                  className={`w-full flex items-center justify-between text-left px-3 py-2.5 rounded-xl text-xs transition-colors ${
                    district === d.district ? "bg-primary/10 border border-primary/30" : "hover:bg-surface/80 border border-transparent"
                  }`}
                >
                  <span className="text-sand truncate mr-2">{d.district}</span>
                  <RiskBadge level={d.predicted_risk} size="sm" />
                </button>
              ))}
            </div>
          </BentoCard>

          <BentoCard>
            <div className="flex items-center gap-1.5 text-muted text-[10px] uppercase tracking-wide mb-2">
              <TrendingUp size={12} />
              {t("assistant.droughtTrend").replace("{district}", district || districts[0]?.district || "")}
            </div>
            <ResponsiveContainer width="100%" height={90}>
              <AreaChart data={trend || []}>
                <defs>
                  <linearGradient id="assistantTrendFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="rgb(var(--c-primary))" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="rgb(var(--c-primary))" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="date" hide />
                <YAxis hide domain={["auto", "auto"]} />
                <Tooltip contentStyle={{ backgroundColor: "rgb(var(--c-card))", border: "1px solid rgb(var(--c-cardBorder))", borderRadius: 12, fontSize: 11 }} />
                <Area type="monotone" dataKey="drought" stroke="rgb(var(--c-primary))" fill="url(#assistantTrendFill)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </BentoCard>

          <BentoCard>
            <SectionLabel>{t("assistant.tryAsking")}</SectionLabel>
            <div className="space-y-2">
              {SUGGESTED.map((q) => (
                <button
                  key={q}
                  onClick={() => send(q)}
                  className="text-left text-xs text-sand bg-surface/60 border border-cardBorder rounded-xl px-3 py-2.5 w-full hover:border-primary/40 hover:bg-primary/5 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </BentoCard>
        </div>

        <div className="lg:col-span-3 bento-card flex flex-col p-0 overflow-hidden min-h-[680px]">
          <div className="border-b border-cardBorder px-5 py-4 flex items-center justify-between gap-3 bg-surface/40">
            <span className="text-xs text-muted">
              {t("assistant.focus")}: <span className="text-sand font-medium">{district || t("assistant.continental")}</span>
            </span>
            <select
              value={district}
              onChange={(e) => focusDistrict(e.target.value)}
              className="input-field w-auto max-w-[220px] py-2 text-xs"
            >
              <option value="">{t("assistant.continental")}</option>
              {districts.map((d) => (
                <option key={d.district} value={d.district}>{d.district}</option>
              ))}
            </select>
          </div>

          <div ref={scrollRef} className="flex-1 overflow-y-auto p-5 md:p-6 space-y-4">
            {briefingLoading && (
              <div className="flex gap-3">
                <Avatar role="assistant" />
                <div className="chat-bubble-assistant">{t("assistant.preparing")}</div>
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`flex gap-3 ${m.role === "user" ? "justify-end" : ""}`}>
                {m.role === "assistant" && <Avatar role="assistant" />}
                <div className={m.role === "user" ? "chat-bubble-user max-w-[85%]" : "chat-bubble-assistant max-w-[85%]"}>
                  {m.text}
                  {m.aiError && (
                    <div className="text-[10px] text-amber mt-2 font-mono">{m.aiError}</div>
                  )}
                </div>
                {m.role === "user" && <Avatar role="user" />}
              </div>
            ))}
            {loading && (
              <div className="flex items-center gap-2 text-muted text-sm pl-12">
                <Loader2 size={14} className="animate-spin text-primary" /> {t("assistant.analyzing")}
              </div>
            )}
          </div>

          <div className="border-t border-cardBorder p-4 flex gap-2 bg-surface/30">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
              placeholder={t("assistant.placeholder")}
              className="input-field flex-1"
            />
            <button onClick={() => send()} disabled={loading} className="btn-primary px-5">
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function Avatar({ role }) {
  if (role === "assistant") {
    return (
      <div className="w-9 h-9 rounded-2xl bg-primary/15 flex items-center justify-center flex-shrink-0">
        <Bot size={16} className="text-primary" />
      </div>
    );
  }
  return (
    <div className="w-9 h-9 rounded-2xl bg-cardBorder/50 flex items-center justify-center flex-shrink-0">
      <User size={16} className="text-muted" />
    </div>
  );
}

function districtCounts(districts) {
  return districts?.reduce((acc, d) => {
    acc[d.predicted_risk] = (acc[d.predicted_risk] || 0) + 1;
    return acc;
  }, {});
}
