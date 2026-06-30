import { useEffect, useState } from "react";
import { Waves, CloudRain, Sun, AlertOctagon } from "lucide-react";
import { simulateScenario } from "../api/client";
import { PageHeader, Card, Metric } from "../components/ui";
import RiskBadge from "../components/RiskBadge";
import AINarrativeCard from "../components/AINarrativeCard";
import DistrictTransitionGrid from "../components/DistrictTransitionGrid";
import { useLanguage } from "../context/LanguageContext";

export default function ScenarioSimulator() {
  const { t, language } = useLanguage();
  const [delta, setDelta] = useState(0);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showTable, setShowTable] = useState(false);

  useEffect(() => {
    setLoading(true);
    const handle = setTimeout(() => {
      simulateScenario(delta, language).then((d) => {
        setResult(d);
        setLoading(false);
      });
    }, 350);
    return () => clearTimeout(handle);
  }, [delta, language]);

  const isDrought = delta < 0;
  const isFlood = delta > 0;
  const intensity = Math.min(Math.abs(delta) / 80, 1);

  return (
    <div className="animate-fade-up">
      <PageHeader
        eyebrow={t("scenario.eyebrow")}
        title={t("scenario.title")}
        subtitle={t("scenario.subtitle")}
      />

      <Card className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <label className="text-muted text-xs uppercase tracking-wide">{t("scenario.sliderLabel")}</label>
          <div className="flex items-center gap-2">
            {isDrought ? (
              <Sun size={18} className="text-clay" style={{ opacity: 0.4 + intensity * 0.6 }} />
            ) : isFlood ? (
              <CloudRain size={18} className="text-primary" style={{ opacity: 0.4 + intensity * 0.6 }} />
            ) : (
              <Waves size={18} className="text-gold" />
            )}
            <span className={`font-mono text-lg font-bold ${isDrought ? "text-clay" : isFlood ? "text-primary" : "text-gold"}`}>
              {delta > 0 ? "+" : ""}{delta}%
            </span>
          </div>
        </div>
        <input
          type="range"
          min={-80}
          max={80}
          step={5}
          value={delta}
          onChange={(e) => setDelta(Number(e.target.value))}
          className="w-full accent-gold"
        />
        <div className="flex justify-between text-muted text-[11px] mt-1.5">
          <span>{t("scenario.severeDrought")}</span>
          <span>{t("scenario.current")}</span>
          <span>{t("scenario.floodScenario")}</span>
        </div>
      </Card>

      <AINarrativeCard narrative={result?.ai_narrative} mode={result?.ai_mode} loading={loading && !result} />

      {result && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
            <Metric label={t("scenario.criticalBefore")} value={result.n_critical_current} sublabel={t("scenario.climateShock")} />
            <Metric
              label={t("scenario.criticalAfter")}
              value={result.n_critical_projected}
              sublabel={`${result.n_critical_projected - result.n_critical_current >= 0 ? "+" : ""}${result.n_critical_projected - result.n_critical_current} · ${t("scenario.climateShock")}`}
            />
            <Metric label={t("scenario.scenarioLabel")} value={`${delta > 0 ? "+" : ""}${delta}%`} sublabel={t("scenario.vsCurrentRainfall")} />
          </div>

          {result.v2_available && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <Metric label={t("scenario.fsCriticalBefore")} value={result.n_fs_critical_current} sublabel={t("scenario.foodSecurity")} accent="primary" />
              <Metric
                label={t("scenario.fsCriticalAfter")}
                value={result.n_fs_critical_projected}
                sublabel={`${result.n_fs_critical_projected - result.n_fs_critical_current >= 0 ? "+" : ""}${result.n_fs_critical_projected - result.n_fs_critical_current} · ${t("scenario.foodSecurity")}`}
                accent="primary"
              />
              <div className="bento-card p-4 flex flex-col justify-center">
                <div className="text-muted text-[10px] font-mono">{t("scenario.bothModelsNote")}</div>
              </div>
            </div>
          )}

          {(result.v2_available
            ? result.n_fs_critical_projected > result.n_fs_critical_current
            : result.n_critical_projected > result.n_critical_current) && (
            <div className="flex items-center gap-2 text-clay bg-clay/10 border border-clay/30 rounded-2xl px-4 py-3 mb-6 text-sm">
              <AlertOctagon size={16} className="flex-shrink-0" />
              {t("scenario.warningWorsens")}
            </div>
          )}

          <div className="text-sand font-display text-sm mb-3">{t("scenario.gridTitle")}</div>
          <DistrictTransitionGrid districts={result.districts} />

          <button
            onClick={() => setShowTable(!showTable)}
            className="text-muted text-xs font-mono underline hover:text-sand transition-colors mb-3"
          >
            {showTable ? t("scenario.hideTable") : t("scenario.showTable")}
          </button>

          {showTable && (
            <Card className="p-0 overflow-hidden mb-6">
              <div className="overflow-y-auto" style={{ maxHeight: 380 }}>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-muted text-xs uppercase border-t border-cardBorder">
                      <th className="text-left px-5 py-2">{t("scenario.district")}</th>
                      <th className="text-left px-3 py-2">{t("scenario.climateShock")}</th>
                      {result.v2_available && <th className="text-left px-3 py-2">{t("scenario.foodSecurity")}</th>}
                      <th className="text-right px-5 py-2">{t("scenario.droughtIndexProj")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.districts.map((d) => (
                      <tr key={d.district} className="border-t border-cardBorder">
                        <td className="px-5 py-2 text-sand">{d.district} <span className="text-muted text-xs">({d.country})</span></td>
                        <td className="px-3 py-2 flex items-center gap-1"><RiskBadge level={d.current_risk} size="sm" /><span className="text-muted text-xs">→</span><RiskBadge level={d.projected_risk} size="sm" /></td>
                        {result.v2_available && (
                          <td className="px-3 py-2 flex items-center gap-1">
                            <RiskBadge level={d.food_security_current_risk} size="sm" /><span className="text-muted text-xs">→</span><RiskBadge level={d.food_security_projected_risk} size="sm" />
                          </td>
                        )}
                        <td className="px-5 py-2 text-right font-mono text-muted">{d.projected_drought_index}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          <p className="text-muted text-xs mt-2 flex items-center gap-1.5">
            <Waves size={12} /> {result.method}
          </p>
        </>
      )}
    </div>
  );
}
