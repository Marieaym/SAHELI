import { ArrowRight, TrendingDown, TrendingUp, Minus, ShieldAlert, Cloud } from "lucide-react";
import { RISK_COLORS } from "../api/client";
import RiskBadge from "./RiskBadge";
import { useLanguage } from "../context/LanguageContext";

const SEVERITY = { Critical: 4, High: 3, Medium: 2, Low: 1 };

export default function DistrictTransitionGrid({ districts }) {
  const { t } = useLanguage();
  const hasV2 = districts.some((d) => d.food_security_current_risk);

  const sorted = [...districts].sort((a, b) => {
    const keyB = b.food_security_projected_risk || b.projected_risk;
    const keyA = a.food_security_projected_risk || a.projected_risk;
    const deltaB = SEVERITY[b.projected_risk] - SEVERITY[b.current_risk];
    const deltaA = SEVERITY[a.projected_risk] - SEVERITY[a.current_risk];
    if (deltaB !== deltaA) return deltaB - deltaA;
    return SEVERITY[keyB] - SEVERITY[keyA];
  });

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-6">
      {sorted.map((d) => {
        const delta = SEVERITY[d.projected_risk] - SEVERITY[d.current_risk];
        const Icon = delta > 0 ? TrendingDown : delta < 0 ? TrendingUp : Minus;
        const trendColor = delta > 0 ? "text-clay" : delta < 0 ? "text-acacia" : "text-muted";
        const disagrees = hasV2 && d.food_security_projected_risk !== d.projected_risk;

        return (
          <div
            key={d.district}
            className="bento-card p-4 relative overflow-hidden transition-transform hover:-translate-y-0.5"
            style={{ borderLeft: `3px solid ${RISK_COLORS[hasV2 ? d.food_security_projected_risk : d.projected_risk]}` }}
          >
            <div className="flex items-center justify-between mb-3">
              <div>
                <div className="font-display font-semibold text-sm text-sand">{d.district}</div>
                <div className="text-muted text-[10px] font-mono">{d.country}</div>
              </div>
              <Icon size={16} className={trendColor} />
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Cloud size={11} className="text-muted flex-shrink-0" />
                <span className="text-muted text-[9px] font-mono uppercase w-14 flex-shrink-0">{t("scenario.climateShock")}</span>
                <RiskBadge level={d.current_risk} size="sm" />
                <ArrowRight size={11} className="text-muted flex-shrink-0" />
                <RiskBadge level={d.projected_risk} size="sm" />
              </div>

              {hasV2 && (
                <div className="flex items-center gap-2">
                  <ShieldAlert size={11} className="text-muted flex-shrink-0" />
                  <span className="text-muted text-[9px] font-mono uppercase w-14 flex-shrink-0">{t("scenario.foodSecurity")}</span>
                  <RiskBadge level={d.food_security_current_risk} size="sm" />
                  <ArrowRight size={11} className="text-muted flex-shrink-0" />
                  <RiskBadge level={d.food_security_projected_risk} size="sm" />
                </div>
              )}
            </div>

            {disagrees && (
              <div className="text-amber text-[9px] font-mono mt-2">{t("scenario.modelsDisagree")}</div>
            )}
            {hasV2 && d.food_security_ground_truth_status === "extrapolated" && (
              <div className="text-muted text-[9px] font-mono mt-1 italic">{t("scenario.extrapolated")}</div>
            )}
            {delta > 0 && !hasV2 && (
              <div className="text-clay text-[10px] font-mono mt-2">{t("scenario.worsens")}</div>
            )}
          </div>
        );
      })}
    </div>
  );
}
