import { useEffect, useState } from "react";
import { ArrowRight, Info, CheckCircle2, XCircle } from "lucide-react";
import { getDistricts, getDistrictDetail, getCausalEffect } from "../api/client";
import { PageHeader, Card, LoadingState, SectionLabel } from "../components/ui";
import RiskBadge from "../components/RiskBadge";
import { useLanguage } from "../context/LanguageContext";

export default function CausalPathway() {
  const { t } = useLanguage();
  const [districts, setDistricts] = useState(null);
  const [selected, setSelected] = useState("");
  const [detail, setDetail] = useState(null);
  const [causal, setCausal] = useState(null);

  useEffect(() => {
    getDistricts().then((d) => {
      setDistricts(d.districts);
      if (d.districts.length) setSelected(d.districts[0].district);
    });
    getCausalEffect().then(setCausal).catch(() => setCausal(null));
  }, []);

  useEffect(() => {
    if (selected) getDistrictDetail(selected).then(setDetail);
  }, [selected]);

  if (!districts) return <LoadingState message={t("overview.loading")} />;

  const severity = detail
    ? detail.drought_index < -0.5 ? "high" : detail.drought_index < -0.2 ? "medium" : "low"
    : "low";
  const measuredColor = severity === "high" ? "#B83227" : severity === "medium" ? "#D9822B" : "#6B9080";

  const nodes = detail ? [
    { id: "rain", label: "Rainfall Deficit", measured: true, value: `${detail.drought_index.toFixed(2)} idx` },
    { id: "water", label: "Soil / Water Balance", measured: true, value: `${detail.water_balance_30d?.toFixed(1) ?? "—"} mm` },
    { id: "conflict", label: "Conflict Activity (ACLED)", measured: true, value: `${Math.round(detail.conflict_events_30d ?? 0)} events/30d` },
    { id: "yield", label: "Crop Yield Pressure", measured: false, value: "illustrative" },
    { id: "price", label: "Market Price Pressure (WFP)", measured: true, value: `${(detail.price_anomaly_30d ?? 0).toFixed(2)}σ` },
    { id: "risk", label: "Food Security Risk", measured: true, value: detail.predicted_risk },
  ] : [];

  return (
    <div className="animate-fade-up">
      <PageHeader
        eyebrow={t("causal.eyebrow")}
        title={t("causal.title")}
        subtitle={t("causal.subtitle")}
      />

      <div className="flex items-start gap-3 bg-amber/10 border border-amber/30 rounded-xl px-4 py-3 mb-6">
        <Info size={18} className="text-amber flex-shrink-0 mt-0.5" />
        <p className="text-sand text-sm leading-relaxed">{t("causal.disclaimer")}</p>
      </div>

      {causal && (
        <Card className="mb-6">
          <div className="flex items-center justify-between mb-3">
            <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-gold">
              {t("causal.validatedEstimates")}
            </span>
          </div>

          <div className="grid md:grid-cols-3 gap-5 mb-2">
            {[
              { key: "drought_effect", label: "drought → Critical risk", color: "text-clay", digits: 1 },
              { key: "conflict_effect", label: "real ACLED conflict → Critical risk", color: "text-acacia", digits: 2 },
              { key: "price_effect", label: "real WFP price shock → Critical risk", color: "text-amber", digits: 2 },
            ].map(({ key, label, color, digits }, i) => {
              const eff = causal[key];
              if (!eff) return null;
              const sign = eff.average_treatment_effect >= 0 ? "+" : "";
              return (
                <div key={key} className={i > 0 ? "border-l border-cardBorder pl-5" : ""}>
                  <div className="flex flex-wrap items-baseline gap-2 mb-1">
                    <span className={`font-display font-semibold text-3xl ${color}`}>
                      {sign}{(eff.average_treatment_effect * 100).toFixed(digits)} pts
                    </span>
                    <span className="text-muted text-xs">{label}</span>
                  </div>
                  <p className="text-sand text-sm leading-relaxed">{eff.interpretation}</p>
                  <div className="flex flex-wrap gap-2 mt-3">
                    {Object.entries(eff.refutation_tests || {}).map(([name, r]) => (
                      <span
                        key={name}
                        className="inline-flex items-center gap-1.5 text-[11px] font-mono px-2 py-1 rounded-[2px] border"
                        style={{
                          color: r.passes ? "#5A6E4C" : "#A53A26",
                          borderColor: r.passes ? "#5A6E4C" : "#A53A26",
                          backgroundColor: r.passes ? "#5A6E4C14" : "#A53A2614",
                        }}
                      >
                        {r.passes ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
                        {name.replace(/_/g, " ")}
                      </span>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>

          <details className="text-xs text-muted mt-4">
            <summary className="cursor-pointer text-gold">{t("causal.limitations")}</summary>
            <ul className="list-disc list-inside mt-2 space-y-1">
              {causal.honest_limitations?.map((l, i) => <li key={i}>{l}</li>)}
            </ul>
          </details>
        </Card>
      )}

      <Card className="mb-6">
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          className="input-field max-w-xs"
        >
          {districts.map((d) => (
            <option key={d.district} value={d.district}>{d.district} — {d.country}</option>
          ))}
        </select>
      </Card>

      {detail && (
        <Card>
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="font-display text-lg text-sand">{detail.district}, {detail.country}</h3>
              <span className="text-muted text-xs">{detail.zone} zone</span>
            </div>
            <RiskBadge level={detail.predicted_risk} />
          </div>

          <div className="flex flex-col md:flex-row items-stretch gap-2">
            {nodes.map((node, i) => (
              <div key={node.id} className="flex items-center gap-2 flex-1">
                <div
                  className={`flex-1 rounded-lg p-4 text-center ${node.measured ? "border-2" : "border border-dashed border-cardBorder"}`}
                  style={node.measured ? { borderColor: measuredColor, backgroundColor: `${measuredColor}15` } : {}}
                >
                  <div className="text-sand text-sm font-medium">{node.label}</div>
                  <div className="font-mono text-xs mt-1" style={{ color: node.measured ? measuredColor : undefined }}>
                    <span className={node.measured ? "" : "text-muted"}>{node.value}</span>
                  </div>
                  <div className="text-[10px] text-muted mt-1 uppercase tracking-wide">
                    {node.measured ? t("causal.measured") : t("causal.illustrative")}
                  </div>
                </div>
                {i < nodes.length - 1 && (
                  <ArrowRight className="text-muted flex-shrink-0 hidden md:block" size={20} />
                )}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
