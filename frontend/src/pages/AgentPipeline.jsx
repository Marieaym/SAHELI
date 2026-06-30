import { useEffect, useState, useRef, useMemo } from "react";
import { Satellite, Brain, MessageCircle, Radio, FileText, Check, Loader2, Play, Download, Workflow, TrendingDown, TrendingUp, Minus } from "lucide-react";
import { getDistricts, downloadBrief, streamPipeline, getForecast } from "../api/client";
import { PageHeader, BentoCard, LoadingState, LiveDot } from "../components/ui";
import PipelineFlowDiagram from "../components/PipelineFlowDiagram";
import RiskBadge from "../components/RiskBadge";
import { useLanguage } from "../context/LanguageContext";

const AGENT_KEYS = ["sentinel", "forecast", "explainer", "alerter", "policywriter"];
const AGENT_ICONS = { sentinel: Satellite, forecast: Brain, explainer: MessageCircle, alerter: Radio, policywriter: FileText };

export default function AgentPipeline() {
  const { t, language } = useLanguage();
  const [districts, setDistricts] = useState(null);
  const [selected, setSelected] = useState("");
  const [running, setRunning] = useState(false);
  const [activeStep, setActiveStep] = useState(-1);
  const [results, setResults] = useState({});
  const [complete, setComplete] = useState(false);
  const [forecast, setForecast] = useState(null);
  const stopRef = useRef(null);

  const agents = useMemo(
    () => AGENT_KEYS.map((key) => ({
      key,
      name: t(`pipeline.agents.${key}.name`),
      role: t(`pipeline.agents.${key}.role`),
      icon: AGENT_ICONS[key],
    })),
    [t]
  );

  useEffect(() => {
    getDistricts().then((d) => {
      setDistricts(d.districts);
      if (d.districts.length) setSelected(d.districts[0].district);
    });
    return () => stopRef.current?.();
  }, []);

  useEffect(() => {
    if (!selected) return;
    setForecast(null);
    getForecast(selected).then(setForecast).catch(() => setForecast(null));
  }, [selected]);

  function runPipeline() {
    setRunning(true);
    setResults({});
    setComplete(false);
    setActiveStep(0);

    stopRef.current = streamPipeline(selected, language, (event) => {
      if (event.step === "error") {
        setRunning(false);
        return;
      }
      if (event.step === "complete") {
        setActiveStep(5);
        setComplete(true);
        setRunning(false);
        return;
      }
      const idx = AGENT_KEYS.indexOf(event.step);
      if (idx >= 0) {
        setResults((r) => ({ ...r, [event.step]: event }));
        setActiveStep(idx + 1);
      }
    });
  }

  if (!districts) return <LoadingState />;
  const data = districtData(districts, selected);

  return (
    <div className="animate-fade-up">
      <PageHeader
        eyebrow={t("pipeline.eyebrow")}
        title={t("pipeline.title")}
        subtitle={t("pipeline.subtitle")}
        badge={running ? <LiveDot label={t("pipeline.running")} /> : null}
      />

      <div className="bento-indigo p-6 md:p-8 mb-6 text-white relative overflow-hidden">
        <div className="absolute inset-0 opacity-20 pointer-events-none">
          <div className="absolute -right-20 -top-20 w-64 h-64 rounded-full bg-white/30 blur-3xl" />
          <div className="absolute -left-10 bottom-0 w-48 h-48 rounded-full bg-cyan-400/40 blur-2xl" />
        </div>
        <div className="relative flex flex-wrap items-center gap-4">
          <div className="w-12 h-12 rounded-2xl bg-white/15 flex items-center justify-center">
            <Workflow size={22} />
          </div>
          <div className="flex-1 min-w-[200px]">
            <div className="text-[10px] uppercase tracking-widest font-mono opacity-70 mb-1">{t("pipeline.selectDistrict")}</div>
            <select
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              disabled={running}
              className="bg-white/10 border border-white/20 text-white rounded-xl px-4 py-2.5 w-full max-w-sm backdrop-blur-sm focus:outline-none focus:ring-2 focus:ring-white/30"
            >
              {districts.map((d) => (
                <option key={d.district} value={d.district} className="text-night">{d.district} — {d.country}</option>
              ))}
            </select>
          </div>
          {data && <RiskBadge level={data.predicted_risk} />}
          <button
            onClick={runPipeline}
            disabled={running}
            className="ml-auto bg-goldBright text-night font-semibold rounded-xl px-6 py-3 flex items-center gap-2 disabled:opacity-50 hover:bg-gold transition-colors shadow-lg"
          >
            {running ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} />}
            {running ? t("pipeline.running") : t("pipeline.run")}
          </button>
        </div>
      </div>

      <PipelineFlowDiagram activeIndex={activeStep} complete={complete} />

      <div className="space-y-4">
        {agents.map((agent, i) => {
          const Icon = agent.icon;
          const status = activeStep > i || activeStep === 5 ? "done" : activeStep === i ? "active" : "pending";
          const result = results[agent.key];

          return (
            <BentoCard
              key={agent.key}
              className={`transition-all duration-300 ${
                status === "active" ? "ring-2 ring-primary/40 shadow-bento-lg" : status === "pending" ? "opacity-55" : ""
              }`}
            >
              <div className="flex items-start gap-4">
                <div
                  className={`w-11 h-11 rounded-2xl flex items-center justify-center flex-shrink-0 ${
                    status === "done" ? "bg-acacia/15" : status === "active" ? "bg-primary/15" : "bg-cardBorder/40"
                  }`}
                >
                  {status === "done" ? (
                    <Check size={20} className="text-acacia" />
                  ) : status === "active" ? (
                    <Loader2 size={20} className="text-primary animate-spin" />
                  ) : (
                    <Icon size={20} className="text-muted" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-display font-semibold text-sand">{agent.name}</span>
                    <span className="text-muted text-xs">· {agent.role}</span>
                  </div>

                  {agent.key === "sentinel" && result && (
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-3">
                      <MetricChip label={t("pipeline.droughtIndex")} value={result.drought_index.toFixed(2)} />
                      <MetricChip label={t("pipeline.dryDays")} value={result.consec_dry_days} />
                      <MetricChip label={t("pipeline.zone")} value={result.zone} />
                    </div>
                  )}

                  {agent.key === "forecast" && result && (
                    <div className="mt-3 flex flex-wrap items-center gap-3">
                      <RiskBadge level={result.predicted_risk} size="sm" />
                      <span className="text-muted text-xs font-mono">
                        P({result.predicted_risk}) = {(result.probabilities[result.predicted_risk.toLowerCase()] * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}

                  {agent.key === "forecast" && result && forecast && (
                    <ForecastHorizonRow forecast={forecast} t={t} />
                  )}

                  {agent.key === "explainer" && result && (
                    <div className="mt-3 bg-surface/80 border border-cardBorder rounded-2xl p-4 text-sm text-sand leading-relaxed">
                      {result.explanation}
                      <div className="text-muted text-[10px] mt-2 font-mono uppercase tracking-wide">{t("pipeline.shapNote")}</div>
                    </div>
                  )}

                  {agent.key === "alerter" && result && (
                    <div className="mt-3 bg-amber-500/15 border border-amber-500/30 text-sand rounded-2xl p-4 text-sm max-w-xl">
                      {result.message}
                    </div>
                  )}

                  {agent.key === "policywriter" && result?.ready && (
                    <button
                      onClick={() => downloadBrief(selected, language)}
                      className="mt-3 inline-flex items-center gap-2 btn-primary text-sm px-4 py-2"
                    >
                      <Download size={15} /> {t("pipeline.downloadBrief")}
                    </button>
                  )}
                </div>
              </div>
            </BentoCard>
          );
        })}
      </div>

      {complete && (
        <BentoCard className="mt-5 border border-acacia/30 bg-acacia/5">
          <p className="text-sand text-sm leading-relaxed">
            {t("pipeline.complete").replace("{district}", selected)}
          </p>
        </BentoCard>
      )}
    </div>
  );
}

function ForecastHorizonRow({ forecast, t }) {
  return (
    <div className="mt-4 bg-surface/80 border border-cardBorder rounded-2xl p-4">
      <div className="text-muted text-[10px] uppercase tracking-wide font-mono mb-3">{t("pipeline.forecastTitle")}</div>
      <div className="flex items-stretch gap-2 overflow-x-auto">
        <HorizonPoint label={t("pipeline.forecastToday")} value={forecast.current_drought_index} isToday />
        {forecast.forecasts.map((f) => (
          <HorizonPoint
            key={f.horizon_weeks}
            label={`+${f.horizon_weeks}w`}
            value={f.forecast_drought_index}
            delta={f.change_from_current}
            t={t}
          />
        ))}
      </div>
      <div className="text-muted text-[10px] mt-3 leading-relaxed">{forecast.note}</div>
    </div>
  );
}

function HorizonPoint({ label, value, delta, isToday, t }) {
  // Lower drought_index = worse conditions in SAHELI's convention.
  const direction = isToday ? null : delta > 0.05 ? "improving" : delta < -0.05 ? "worsening" : "stable";
  const Icon = direction === "improving" ? TrendingUp : direction === "worsening" ? TrendingDown : Minus;
  const colorClass =
    direction === "improving" ? "text-acacia" : direction === "worsening" ? "text-clay" : "text-muted";

  return (
    <div
      className={`flex-1 min-w-[88px] rounded-xl px-3 py-2.5 text-center ${
        isToday ? "bg-primary/10 border border-primary/25" : "bg-card border border-cardBorder"
      }`}
    >
      <div className="font-mono text-[9px] uppercase tracking-wide text-muted">{label}</div>
      <div className="font-display font-bold text-lg text-sand mt-1">{value.toFixed(2)}</div>
      {!isToday && (
        <div className={`flex items-center justify-center gap-1 text-[9px] font-mono mt-1 ${colorClass}`}>
          <Icon size={10} />
          {direction === "improving" ? t("pipeline.forecastImproving") : direction === "worsening" ? t("pipeline.forecastWorsening") : t("pipeline.forecastStable")}
        </div>
      )}
    </div>
  );
}

function districtData(districts, selected) {
  return districts.find((d) => d.district === selected);
}

function MetricChip({ label, value }) {
  return (
    <div className="bg-surface/80 border border-cardBorder rounded-xl px-4 py-3">
      <div className="text-muted text-[10px] uppercase tracking-wide font-mono">{label}</div>
      <div className="font-mono text-sand font-semibold mt-1">{value}</div>
    </div>
  );
}
