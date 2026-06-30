import { useEffect, useState } from "react";
import { Wallet, ShieldCheck, ShieldAlert, AlertTriangle } from "lucide-react";
import { simulateIntervention } from "../api/client";
import { PageHeader, Card, Metric } from "../components/ui";
import AINarrativeCard from "../components/AINarrativeCard";
import AllocationBars from "../components/AllocationBars";
import RiskBadge from "../components/RiskBadge";
import { useLanguage } from "../context/LanguageContext";

function formatUSD(n) {
  return `$${Math.round(n).toLocaleString()}`;
}

export default function InterventionSimulator() {
  const { t, language } = useLanguage();
  const [budget, setBudget] = useState(1_500_000);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showTable, setShowTable] = useState(false);

  useEffect(() => {
    setLoading(true);
    const handle = setTimeout(() => {
      simulateIntervention(budget, language).then((d) => {
        setResult(d);
        setLoading(false);
      });
    }, 300);
    return () => clearTimeout(handle);
  }, [budget, language]);

  const resolvedCount = result ? Math.max(result.n_critical_before - result.n_critical_after, 0) : 0;
  const nDisagree = result?.model_disagreement_districts?.length || 0;

  return (
    <div className="animate-fade-up">
      <PageHeader
        eyebrow={t("intervention.eyebrow")}
        title={t("intervention.title")}
        subtitle={t("intervention.subtitle")}
      />

      <Card className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <label className="text-muted text-xs uppercase tracking-wide">{t("intervention.sliderLabel")}</label>
          <div className="flex items-center gap-2">
            <Wallet size={18} className="text-gold" />
            <span className="font-mono text-lg font-bold text-gold">{formatUSD(budget)}</span>
          </div>
        </div>
        <input
          type="range"
          min={100000}
          max={5000000}
          step={100000}
          value={budget}
          onChange={(e) => setBudget(Number(e.target.value))}
          className="w-full accent-gold"
        />
        <div className="flex justify-between text-muted text-[11px] mt-1.5">
          <span>$100K</span>
          <span>$5M</span>
        </div>
      </Card>

      {result?.v2_available && (
        <div className="flex items-center gap-2 text-primary bg-primary/10 border border-primary/30 rounded-2xl px-4 py-2.5 mb-4 text-xs font-mono">
          <ShieldAlert size={14} className="flex-shrink-0" />
          {t("intervention.basedOnRealModel")}
        </div>
      )}

      <AINarrativeCard narrative={result?.ai_narrative} mode={result?.ai_mode} loading={loading && !result} />

      {result && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <Metric label={t("intervention.criticalBefore")} value={result.n_critical_before} sublabel={result.v2_available ? t("scenario.foodSecurity") : null} accent="primary" />
            <Metric
              label={t("intervention.criticalAfter")}
              value={result.n_critical_after}
              sublabel={`${result.n_critical_after - result.n_critical_before >= 0 ? "+" : ""}${result.n_critical_after - result.n_critical_before}`}
              accent="primary"
            />
            <Metric label={t("intervention.avgPerCritical")} value={formatUSD(budget / Math.max(result.n_critical_before, 1))} />
          </div>

          {resolvedCount > 0 && (
            <div className="flex items-center gap-2 text-acacia bg-acacia/10 border border-acacia/30 rounded-2xl px-4 py-3 mb-4 text-sm">
              <ShieldCheck size={16} className="flex-shrink-0" />
              {t("intervention.resolvedNote").replace("{n}", resolvedCount)}
            </div>
          )}

          {nDisagree > 0 && (
            <div className="flex items-center gap-2 text-amber bg-amber/10 border border-amber/30 rounded-2xl px-4 py-3 mb-6 text-sm">
              <AlertTriangle size={16} className="flex-shrink-0" />
              {t("intervention.disagreeNote").replace("{n}", nDisagree).replace("{list}", result.model_disagreement_districts.join(", "))}
            </div>
          )}

          <div className="text-sand font-display text-sm mb-3">{t("intervention.allocationTitle")}</div>
          <AllocationBars allocations={result.allocations} />

          <button
            onClick={() => setShowTable(!showTable)}
            className="text-muted text-xs font-mono underline hover:text-sand transition-colors mb-3"
          >
            {showTable ? t("scenario.hideTable") : t("scenario.showTable")}
          </button>

          {showTable && (
            <Card className="p-0 overflow-hidden mb-6">
              <div className="overflow-y-auto" style={{ maxHeight: 420 }}>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-muted text-xs uppercase border-t border-cardBorder">
                      <th className="text-left px-5 py-2">{t("scenario.district")}</th>
                      <th className="text-left px-3 py-2">{t("scenario.climateShock")}</th>
                      {result.v2_available && <th className="text-left px-3 py-2">{t("scenario.foodSecurity")}</th>}
                      <th className="text-right px-3 py-2">{t("intervention.allocated")}</th>
                      <th className="text-left px-5 py-2">{t("scenario.projected")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...result.allocations].sort((a, b) => b.allocation - a.allocation).map((row) => (
                      <tr key={row.district} className="border-t border-cardBorder">
                        <td className="px-5 py-2 text-sand">{row.district} <span className="text-muted text-xs">({row.country})</span></td>
                        <td className="px-3 py-2"><RiskBadge level={row.climate_shock_risk || row.predicted_risk} size="sm" /></td>
                        {result.v2_available && (
                          <td className="px-3 py-2"><RiskBadge level={row.food_security_risk} size="sm" /></td>
                        )}
                        <td className="px-3 py-2 text-right font-mono text-gold">{formatUSD(row.allocation)}</td>
                        <td className="px-5 py-2"><RiskBadge level={row.projected_risk} size="sm" /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
