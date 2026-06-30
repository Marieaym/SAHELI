import { ArrowRight, Cloud, ShieldAlert } from "lucide-react";
import RiskBadge from "./RiskBadge";
import { useLanguage } from "../context/LanguageContext";

function formatUSD(n) {
  return `$${Math.round(n).toLocaleString()}`;
}

export default function AllocationBars({ allocations }) {
  const { t } = useLanguage();
  const hasV2 = allocations.some((d) => d.food_security_risk);
  const sorted = [...allocations].sort((a, b) => b.allocation - a.allocation);
  const maxAlloc = Math.max(...sorted.map((d) => d.allocation), 1);

  return (
    <div className="space-y-3 mb-6">
      {sorted.map((d) => {
        const disagrees = hasV2 && d.food_security_risk !== d.climate_shock_risk;
        return (
          <div key={d.district} className="bento-card p-4">
            <div className="flex items-center justify-between mb-2 flex-wrap gap-1.5">
              <div className="flex items-center gap-2 min-w-0">
                <span className="font-display font-semibold text-sm text-sand truncate">{d.district}</span>
                <span className="text-muted text-[10px] font-mono flex-shrink-0">{d.country}</span>
                {disagrees && (
                  <span className="text-amber text-[9px] font-mono border border-amber/40 bg-amber/10 rounded px-1 flex-shrink-0">
                    {t("scenario.modelsDisagree")}
                  </span>
                )}
              </div>
              <div className="flex flex-col items-end gap-1 flex-shrink-0">
                <div className="flex items-center gap-1.5">
                  <Cloud size={10} className="text-muted" />
                  <RiskBadge level={d.climate_shock_risk || d.predicted_risk} size="sm" />
                </div>
                {hasV2 && (
                  <div className="flex items-center gap-1.5">
                    <ShieldAlert size={10} className="text-muted" />
                    <RiskBadge level={d.food_security_risk} size="sm" />
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex-1 h-2.5 rounded-full bg-cardBorder/40 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-700 ease-out"
                  style={{
                    width: `${(d.allocation / maxAlloc) * 100}%`,
                    background: "linear-gradient(90deg, rgb(var(--c-primary)), rgb(var(--c-goldBright)))",
                  }}
                />
              </div>
              <div className="font-mono text-xs text-goldBright w-24 text-right flex-shrink-0">
                {formatUSD(d.allocation)}
              </div>
            </div>
            {hasV2 && d.food_security_status === "extrapolated" && (
              <div className="text-muted text-[9px] font-mono mt-1.5 italic">{t("scenario.extrapolated")}</div>
            )}
          </div>
        );
      })}
    </div>
  );
}
