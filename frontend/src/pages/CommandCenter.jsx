import { useEffect, useState } from "react";
import {
  Crosshair, Cloud, ShieldAlert, TrendingUp, TrendingDown, Minus,
  AlertTriangle, Leaf, Sparkles, ChevronDown, Loader2, CheckCircle2,
} from "lucide-react";
import { getCommandCenter, getCommandCenterBriefing } from "../api/client";
import { PageHeader, Card, LoadingState, Metric } from "../components/ui";
import RiskBadge from "../components/RiskBadge";
import { useLanguage } from "../context/LanguageContext";

const TREND_ICON = { worsening: TrendingDown, improving: TrendingUp, stable: Minus };
const TREND_COLOR = { worsening: "text-clay", improving: "text-acacia", stable: "text-muted" };

export default function CommandCenter() {
  const { t, language } = useLanguage();
  const [data, setData] = useState(null);
  const [selected, setSelected] = useState(null);
  const [briefing, setBriefing] = useState(null);
  const [briefingLoading, setBriefingLoading] = useState(false);

  useEffect(() => {
    getCommandCenter().then(setData);
  }, []);

  function selectDistrict(district) {
    if (selected === district) { setSelected(null); setBriefing(null); return; }
    setSelected(district);
    setBriefing(null);
    setBriefingLoading(true);
    getCommandCenterBriefing(district, language)
      .then(setBriefing)
      .finally(() => setBriefingLoading(false));
  }

  if (!data) return <LoadingState message={t("commandCenter.loading")} />;

  const selectedRow = data.districts.find((d) => d.district === selected);

  return (
    <div className="animate-fade-up">
      <PageHeader
        eyebrow={t("commandCenter.eyebrow")}
        title={t("commandCenter.title")}
        subtitle={t("commandCenter.subtitle")}
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Metric label={t("commandCenter.districtsTracked")} value={data.n_districts} />
        <Metric label={t("commandCenter.modelsDisagreeCount")} value={data.n_models_disagree} accent={data.n_models_disagree > 0 ? "critical" : "success"} />
        <div className="bento-card p-5 flex items-center gap-2">
          <Sparkles size={16} className="text-primary flex-shrink-0" />
          <span className="text-muted text-xs leading-relaxed">{t("commandCenter.scoringNote")}</span>
        </div>
      </div>

      <div className="space-y-3">
        {data.districts.map((d, i) => {
          const TrendIcon = TREND_ICON[d.forecast_trend] || Minus;
          const isOpen = selected === d.district;
          return (
            <div key={d.district}>
              <button
                onClick={() => selectDistrict(d.district)}
                className="w-full text-left bento-card p-4 hover:-translate-y-0.5 transition-transform"
              >
                <div className="flex items-center gap-4 flex-wrap">
                  <div className="flex items-center gap-3 flex-shrink-0">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center font-mono text-xs font-bold text-primary">
                      {i + 1}
                    </div>
                    <div>
                      <div className="font-display font-semibold text-sm text-sand">{d.district}</div>
                      <div className="text-muted text-[10px] font-mono">{d.country} · {d.zone}</div>
                    </div>
                  </div>

                  <div className="flex items-center gap-1.5">
                    <Cloud size={12} className="text-muted" />
                    <RiskBadge level={d.climate_risk} size="sm" />
                  </div>

                  {d.food_security && (
                    <div className="flex items-center gap-1.5">
                      <ShieldAlert size={12} className="text-muted" />
                      <RiskBadge level={d.food_security.risk} size="sm" />
                      {d.food_security.status === "extrapolated" && (
                        <span className="text-muted text-[9px] font-mono italic">~</span>
                      )}
                    </div>
                  )}

                  {!d.models_agree && (
                    <span className="text-amber text-[9px] font-mono border border-amber/40 bg-amber/10 rounded px-1.5 py-0.5">
                      {t("scenario.modelsDisagree")}
                    </span>
                  )}

                  {d.anomaly === "adverse" && (
                    <div className="flex items-center gap-1 text-clay text-[9px] font-mono">
                      <AlertTriangle size={11} /> {t("commandCenter.anomaly")}
                    </div>
                  )}

                  <div className={`flex items-center gap-1 text-[9px] font-mono ${TREND_COLOR[d.forecast_trend] || "text-muted"}`}>
                    <TrendIcon size={11} /> {d.forecast_trend || "—"}
                  </div>

                  {d.crop_reports.n_total > 0 && (
                    <div className="flex items-center gap-1 text-muted text-[9px] font-mono">
                      <Leaf size={11} /> {d.crop_reports.n_total}
                    </div>
                  )}

                  <div className="ml-auto flex items-center gap-2 flex-shrink-0">
                    <div className="text-right">
                      <div className="font-mono text-[9px] text-muted uppercase">{t("commandCenter.urgency")}</div>
                      <div className="font-display font-bold text-sand text-sm">{d.composite_urgency_score}</div>
                    </div>
                    <ChevronDown size={14} className={`text-muted transition-transform ${isOpen ? "rotate-180" : ""}`} />
                  </div>
                </div>
              </button>

              {isOpen && (
                <Card className="mt-2 mb-1 animate-fade-up">
                  <div className="flex items-start gap-3 mb-4 pb-4 border-b border-cardBorder">
                    <Sparkles size={16} className="text-primary flex-shrink-0 mt-0.5" />
                    {briefingLoading ? (
                      <div className="flex items-center gap-2 text-muted text-sm">
                        <Loader2 size={14} className="animate-spin" /> {t("commandCenter.synthesizing")}
                      </div>
                    ) : (
                      <p className="text-sand/90 text-sm leading-relaxed">{briefing?.briefing}</p>
                    )}
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                    <div>
                      <div className="text-muted font-mono text-[10px] uppercase">{t("commandCenter.droughtIdx")}</div>
                      <div className="text-sand font-mono mt-0.5">{d.drought_index}</div>
                    </div>
                    {d.food_security && (
                      <div>
                        <div className="text-muted font-mono text-[10px] uppercase">IPC</div>
                        <div className="text-sand font-mono mt-0.5">{d.food_security.ipc} / 5</div>
                      </div>
                    )}
                    {d.forecast_8w_drought_index != null && (
                      <div>
                        <div className="text-muted font-mono text-[10px] uppercase">{t("commandCenter.forecast8w")}</div>
                        <div className="text-sand font-mono mt-0.5">{d.forecast_8w_drought_index}</div>
                      </div>
                    )}
                    {d.top_driver && (
                      <div>
                        <div className="text-muted font-mono text-[10px] uppercase">{t("commandCenter.topDriver")}</div>
                        <div className="text-sand mt-0.5">{d.top_driver.name}</div>
                      </div>
                    )}
                  </div>

                  {d.food_security?.status === "validated" && (
                    <div className="flex items-center gap-1.5 text-acacia text-[10px] font-mono mt-3">
                      <CheckCircle2 size={11} /> {t("commandCenter.validatedNote")}
                    </div>
                  )}
                </Card>
              )}
            </div>
          );
        })}
      </div>

      <p className="text-muted text-xs mt-5 leading-relaxed">{t("commandCenter.footerNote")}</p>
    </div>
  );
}
